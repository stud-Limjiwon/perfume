"""온라인 시험 감독 화면 (Tkinter).

화면 1  시험 시작 전: 학생 이름, 시험 시간, 감독관 패스워드 설정
보정    5점 보정: 전체 화면에 점을 하나씩 띄워 집중 구역을 정한다
화면 2  시험 중: 남은 시간, 웹캠, 동공 이탈·자리 이탈 누적 게이지와 횟수
화면 3  시험 종료 후: 패스워드 확인 → 무결성 결과와 의심 정황 목록, 체인값 조회

OpenCV 창은 한글을 쓸 수 없어 화면은 Tkinter로 만들고 웹캠 영상만 그 안에 넣는다.
영상과 얼굴 이미지는 저장하지 않는다.
"""

from __future__ import annotations

import sys
import tkinter as tk
from time import monotonic_ns
from tkinter import font as tkfont
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from config.settings import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PASSWORD_MIN_LENGTH,
    TARGET_FPS,
)
from src.attention_detection import (
    DepartureDetector,
    DepartureKind,
    DetectionCriteria,
    KindSummary,
)
from src.calibration import CalibrationError, Calibrator
from src.features import pupil_sample
from src.landmarks import IRIS_CENTER_INDICES
from src.report import RED, build_report
from src.session_log import (
    PasswordError,
    SessionRecorder,
    list_sessions,
    open_session,
)

# 색 (밝은 배경 기준, 글자 대비 4.5:1 이상)
BG = "#F4F5F7"
PANEL = "#FFFFFF"
LINE = "#DADDE3"
TEXT = "#1D2127"
MUTED = "#5A6270"
ACCENT = "#1F5FD6"
ACCENT_DARK = "#174BAA"
TRACK = "#E6E8EC"
OK = "#1B7A4B"
OK_BG = "#E5F4EC"
RED_TEXT = "#B42318"
RED_BG = "#FDECEA"
YELLOW_TEXT = "#8A5300"
YELLOW_BG = "#FFF3D1"
YELLOW_BAR = "#E0A100"
RED_BAR = "#D92D20"

EXAM_TIME_DEFAULT = "00:05:00"
VIDEO_WIDTH = 640


def parse_duration(text: str) -> int:
    """'HH:MM:SS' → 초. 형식이 틀리거나 0이면 ValueError."""
    parts = text.strip().split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError("시험 시간은 00:00:00 형식으로 입력하세요.")
    hours, minutes, seconds = (int(p) for p in parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("분과 초는 59 이하로 입력하세요.")
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError("시험 시간은 0초보다 길어야 합니다.")
    return total


def format_clock(seconds: float) -> str:
    seconds = max(int(seconds + 0.999), 0)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _pick_font(root: tk.Tk) -> str:
    families = set(tkfont.families(root))
    for name in (
        "Malgun Gothic",
        "맑은 고딕",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Noto Sans CJK KR",
        "NanumGothic",
    ):
        if name in families:
            return name
    return "TkDefaultFont"


class CameraSource:
    """웹캠 프레임과 단조 증가 밀리초 시각을 함께 준다."""

    def __init__(self) -> None:
        self._capture = cv2.VideoCapture(CAMERA_INDEX)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(
                "카메라를 열 수 없습니다. 연결 상태, 다른 앱의 카메라 사용, "
                "카메라 권한을 확인하세요."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self._started_ns = monotonic_ns()
        self._last_ms = -1

    def read(self):
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("카메라에서 영상을 읽지 못했습니다.")
        now = max((monotonic_ns() - self._started_ns) // 1_000_000, self._last_ms + 1)
        self._last_ms = now
        return frame, now

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class Gauge(tk.Canvas):
    """0~1 막대. 70% 이상 노랑, 100% 빨강."""

    def __init__(self, parent, height: int = 14) -> None:
        super().__init__(parent, height=height, bg=TRACK, highlightthickness=0, bd=0)
        self._height = height
        self._ratio = 0.0
        self._bar = self.create_rectangle(0, 0, 0, height, fill=ACCENT, width=0)
        self.bind("<Configure>", lambda _e: self._draw())

    def set(self, ratio: float) -> None:
        self._ratio = max(0.0, min(ratio, 1.0))
        self._draw()

    def _draw(self) -> None:
        width = self.winfo_width()
        color = (
            RED_BAR
            if self._ratio >= 1
            else YELLOW_BAR
            if self._ratio >= 0.7
            else ACCENT
        )
        self.coords(self._bar, 0, 0, width * self._ratio, self._height)
        self.itemconfigure(self._bar, fill=color)


class DepartureCard(tk.Frame):
    """화면 2 오른쪽: 한 종류 이탈의 누적 시간 게이지와 횟수."""

    def __init__(self, parent, app: ExamApp, title: str) -> None:
        super().__init__(
            parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1
        )
        f = app.font
        inner = tk.Frame(self, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=20, pady=12)
        top = tk.Frame(inner, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=title, font=(f, 15, "bold"), bg=PANEL, fg=TEXT).pack(
            side="left"
        )
        self.state = tk.Label(top, text="", font=(f, 11, "bold"), bg=PANEL)
        self.state.pack(side="right")

        tk.Label(
            inner, text="누적 이탈 시간", font=(f, 11), bg=PANEL, fg=MUTED
        ).pack(anchor="w", pady=(8, 0))
        self.time_value = tk.Label(inner, font=(f, 20, "bold"), bg=PANEL, fg=TEXT)
        self.time_value.pack(anchor="w")
        self.time_gauge = Gauge(inner)
        self.time_gauge.pack(fill="x", pady=(4, 0))

        tk.Label(inner, text="이탈 횟수", font=(f, 11), bg=PANEL, fg=MUTED).pack(
            anchor="w", pady=(8, 0)
        )
        self.count_value = tk.Label(inner, font=(f, 20, "bold"), bg=PANEL, fg=TEXT)
        self.count_value.pack(anchor="w")
        self.count_gauge = Gauge(inner)
        self.count_gauge.pack(fill="x", pady=(4, 0))

    def update_values(
        self, summary: KindSummary, active: bool, current_ms: int
    ) -> None:
        c = summary.criteria
        self.time_value.configure(
            text=f"{summary.total_ms / 1000:.1f}초  / {c.suspicious_total_ms / 1000:.0f}초"
        )
        self.count_value.configure(
            text=f"{summary.count}회  / {c.suspicious_count}회"
        )
        self.time_gauge.set(summary.time_ratio)
        self.count_gauge.set(summary.count_ratio)
        if active:
            self.state.configure(text=f"이탈 중 {current_ms / 1000:.1f}초", fg=RED_TEXT)
        elif summary.suspicious:
            self.state.configure(text="의심 기준 도달", fg=RED_TEXT)
        else:
            self.state.configure(text="정상", fg=OK)


class ExamApp:
    def __init__(self, root: tk.Tk, landmark_detector) -> None:
        self.root = root
        self.landmarks = landmark_detector
        self.font = _pick_font(root)
        self.camera: CameraSource | None = None
        self.mode = "setup"
        self.photo = None
        self.fps = 0.0
        self._last_tick_ms: int | None = None
        self.recorder: SessionRecorder | None = None
        self.detector: DepartureDetector | None = None
        self.session_dir = None
        self.final_chain: str | None = None

        root.title("온라인 시험 감독")
        root.configure(bg=BG)
        scale = float(root.tk.call("tk", "scaling")) / (96 / 72)
        root.minsize(round(1000 * scale), round(640 * scale))
        try:
            root.state("zoomed")  # Windows: 최대화
        except tk.TclError:
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        style = ttk.Style(root)
        style.configure("TCombobox", padding=6)
        self.container = tk.Frame(root, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.show_setup()

    # ---------- 공통 ----------
    def _clear(self) -> None:
        self.root.attributes("-fullscreen", False)
        self.root.unbind("<Escape>")
        for child in self.container.winfo_children():
            child.destroy()

    def _button(self, parent, text, command, primary=True, **pack):
        bg, fg, active = (
            (ACCENT, "#FFFFFF", ACCENT_DARK) if primary else (PANEL, TEXT, TRACK)
        )
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=(self.font, 12, "bold"),
            bg=bg,
            fg=fg,
            activebackground=active,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=22,
            pady=11,
            cursor="hand2",
            highlightthickness=0 if primary else 1,
            highlightbackground=LINE,
        )
        return button

    def _field(self, parent, label, show=None, value=""):
        tk.Label(
            parent, text=label, font=(self.font, 12, "bold"), bg=PANEL, fg=TEXT
        ).pack(anchor="w", pady=(12, 4))
        entry = tk.Entry(
            parent,
            font=(self.font, 14),
            show=show or "",
            relief="flat",
            bg="#FAFBFC",
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor=ACCENT,
            insertbackground=TEXT,
        )
        entry.insert(0, value)
        entry.pack(fill="x", ipady=6)
        return entry

    def _stop_camera(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None

    def _fail(self, message: str) -> None:
        self._stop_camera()
        self.mode = "setup"
        messagebox.showerror("오류", message, parent=self.root)
        self.show_setup()

    def on_close(self) -> None:
        if self.mode == "exam":
            if not messagebox.askyesno(
                "시험 진행 중",
                "지금 닫으면 이 시험은 '비정상 종료'로 기록됩니다.\n그래도 닫을까요?",
                parent=self.root,
            ):
                return
        self.mode = "closed"
        self._stop_camera()
        self.root.destroy()

    # ---------- 화면 1: 시험 시작 전 ----------
    def show_setup(self) -> None:
        self._clear()
        self.mode = "setup"
        f = self.font
        wrap = tk.Frame(self.container, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            wrap, text="시험 시작 전입니다", font=(f, 26, "bold"), bg=BG, fg=TEXT
        ).pack(anchor="w")
        tk.Label(
            wrap,
            text="감독관님은 시험 시간과 패스워드를 설정해 주세요.",
            font=(f, 13),
            bg=BG,
            fg=MUTED,
        ).pack(anchor="w", pady=(6, 18))

        card = tk.Frame(wrap, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        card.pack(fill="x")
        body = tk.Frame(card, bg=PANEL, width=520)
        body.pack(padx=32, pady=(14, 28), fill="x")
        self.name_entry = self._field(body, "학생 이름")
        self.time_entry = self._field(body, "시험 시간 (시:분:초)", value=EXAM_TIME_DEFAULT)
        self.pw_entry = self._field(
            body, f"감독관 패스워드 ({PASSWORD_MIN_LENGTH}자 이상)", show="●"
        )
        self.pw2_entry = self._field(body, "패스워드 확인", show="●")
        tk.Label(
            body,
            text="패스워드는 기록을 잠그는 체인값의 키와 결과 조회 본인 확인에 쓰입니다.\n"
            "어디에도 저장되지 않으니 잊지 마세요.",
            font=(f, 11),
            bg=PANEL,
            fg=MUTED,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))
        self.setup_error = tk.Label(body, text="", font=(f, 11, "bold"), bg=PANEL, fg=RED_TEXT)
        self.setup_error.pack(anchor="w", pady=(8, 0))

        buttons = tk.Frame(wrap, bg=BG)
        buttons.pack(fill="x", pady=(18, 0))
        self._button(buttons, "보정 시작", self.start_calibration).pack(side="right")
        self._button(
            buttons, "지난 기록 조회", lambda: self.show_result_auth(None), primary=False
        ).pack(side="left")
        self.name_entry.focus_set()
        self.root.bind("<Return>", lambda _e: self.mode == "setup" and self.start_calibration())

    def start_calibration(self) -> None:
        try:
            name = self.name_entry.get().strip()
            if not name:
                raise ValueError("학생 이름을 입력하세요.")
            exam_seconds = parse_duration(self.time_entry.get())
            password = self.pw_entry.get()
            if len(password) < PASSWORD_MIN_LENGTH:
                raise ValueError(f"패스워드는 {PASSWORD_MIN_LENGTH}자 이상이어야 합니다.")
            if password != self.pw2_entry.get():
                raise ValueError("패스워드 확인이 일치하지 않습니다.")
        except ValueError as error:
            self.setup_error.configure(text=str(error))
            return
        self.pending = {"student": name, "exam_seconds": exam_seconds, "password": password}
        try:
            self.camera = CameraSource()
        except RuntimeError as error:
            self.setup_error.configure(text=str(error))
            return
        self.root.unbind("<Return>")
        self.show_calibration()

    # ---------- 5점 보정 ----------
    def show_calibration(self) -> None:
        self._clear()
        self.mode = "calibration"
        self.calibrator = Calibrator()
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _e: self._cancel_calibration())
        self.calib_canvas = tk.Canvas(self.container, bg=BG, highlightthickness=0)
        self.calib_canvas.pack(fill="both", expand=True)
        self.calib_face = True
        self.root.after(1, self._tick)

    def _cancel_calibration(self) -> None:
        self._stop_camera()
        self.show_setup()

    def _draw_calibration(self) -> None:
        canvas, cal, f = self.calib_canvas, self.calibrator, self.font
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if cal.target is None or width < 10:
            return
        step = cal.index + 1
        canvas.create_text(
            width / 2,
            height / 2 - 90 if cal.index else height * 0.3,
            text=f"고개는 그대로 두고 눈으로 점을 바라보세요  ({step}/{len(cal.targets)})",
            font=(f, 18, "bold"),
            fill=TEXT,
        )
        notice = "얼굴이 보이지 않습니다. 카메라 정면에 앉아 주세요." if not self.calib_face else (
            "이 점을 다시 봐 주세요." if cal.retries else "Esc: 취소"
        )
        canvas.create_text(
            width / 2,
            (height / 2 - 56) if cal.index else height * 0.3 + 34,
            text=notice,
            font=(f, 12),
            fill=RED_TEXT if not self.calib_face else MUTED,
        )
        x, y = cal.target[0] * width, cal.target[1] * height
        canvas.create_oval(x - 26, y - 26, x + 26, y + 26, outline=LINE, width=4)
        canvas.create_arc(
            x - 26, y - 26, x + 26, y + 26,
            start=90, extent=-359.9 * cal.progress, style="arc", outline=ACCENT, width=4,
        )
        canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill=RED_BAR, outline="")

    def _finish_calibration(self, now_ms: int) -> None:
        try:
            zone = self.calibrator.zone()
        except CalibrationError as error:
            messagebox.showwarning("보정 다시 하기", str(error), parent=self.root)
            self.calibrator = Calibrator()
            return
        p = self.pending
        try:
            self.recorder = SessionRecorder(
                student=p["student"],
                password=p["password"],
                exam_seconds=p["exam_seconds"],
                origin_ms=now_ms,
                criteria=DetectionCriteria(),
            )
        except (OSError, ValueError) as error:
            self._fail(f"기록 폴더를 만들 수 없습니다: {error}")
            return
        self.pending = None
        self.session_dir = self.recorder.directory
        self.detector = DepartureDetector(zone, self.recorder.criteria)
        self.origin_ms = now_ms
        self.exam_ms = self.recorder.header["exam_seconds"] * 1000
        self.show_exam()

    # ---------- 화면 2: 시험 중 ----------
    def show_exam(self) -> None:
        self._clear()
        self.mode = "exam"
        f = self.font
        page = tk.Frame(self.container, bg=BG)
        page.pack(fill="both", expand=True, padx=28, pady=20)

        header = tk.Frame(page, bg=BG)
        header.pack(fill="x")
        tk.Label(
            header,
            text=f"응시자  {self.recorder.header['student']}",
            font=(f, 12),
            bg=BG,
            fg=MUTED,
        ).pack(side="left")
        self._button(
            header, "시험 종료 (감독관)", self._ask_end, primary=False
        ).pack(side="right")
        self.fps_label = tk.Label(header, text="", font=(f, 11), bg=BG, fg=MUTED)
        self.fps_label.pack(side="right", padx=(0, 16))
        center = tk.Frame(header, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(center, text="남은 시험 시간", font=(f, 11), bg=BG, fg=MUTED).pack()
        self.clock = tk.Label(center, text="", font=(f, 30, "bold"), bg=BG, fg=TEXT)
        self.clock.pack()
        header.configure(height=86)
        header.pack_propagate(False)

        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True, pady=(12, 0))
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        video_box = tk.Frame(left, bg="#000000")
        video_box.pack()
        self.video = tk.Label(video_box, bg="#000000", width=VIDEO_WIDTH, height=480)
        self.video.pack()
        self.status = tk.Label(
            left, text="", font=(f, 13, "bold"), bg=OK_BG, fg=OK, anchor="w", padx=14, pady=10
        )
        self.status.pack(fill="x", pady=(10, 0))
        tk.Label(
            left,
            text="영상은 저장되지 않습니다. 이탈 시각과 시간만 기록됩니다.",
            font=(f, 10),
            bg=BG,
            fg=MUTED,
        ).pack(anchor="w", pady=(8, 0))

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(22, 0))
        self.gaze_card = DepartureCard(right, self, "동공 이탈")
        self.gaze_card.pack(fill="both", expand=True)
        self.seat_card = DepartureCard(right, self, "자리 이탈")
        self.seat_card.pack(fill="both", expand=True, pady=(14, 0))

    def _ask_end(self) -> None:
        password = simpledialog.askstring(
            "시험 종료", "감독관 패스워드를 입력하세요.", show="●", parent=self.root
        )
        if password is None:
            return
        if not self.recorder.check_password(password):
            messagebox.showerror("시험 종료", "패스워드가 맞지 않습니다.", parent=self.root)
            return
        self._end_exam(self._last_tick_ms)

    def _end_exam(self, now_ms: int | None) -> None:
        if self.mode != "exam":
            return
        self.mode = "ending"
        event = self.detector.finish(now_ms)
        if event is not None:
            self.recorder.record(event)
        self.final_chain = self.recorder.close(self.detector.summary())
        self._stop_camera()
        self.show_result_auth(self.session_dir)

    def _update_exam(self, frame, result, now_ms: int) -> None:
        sample = pupil_sample(result, now_ms, frame.shape[1], frame.shape[0])
        event = self.detector.update(sample)
        if event is not None:
            self.recorder.record(event)
        summary = self.detector.summary(now_ms)
        remaining = (self.exam_ms - (now_ms - self.origin_ms)) / 1000
        self.clock.configure(text=format_clock(remaining), fg=RED_TEXT if remaining <= 60 else TEXT)
        active = summary.current_kind
        out_ms = summary.current_out_ms
        self.gaze_card.update_values(summary.gaze, active is DepartureKind.GAZE, out_ms)
        self.seat_card.update_values(summary.seat, active is DepartureKind.SEAT, out_ms)
        if active is DepartureKind.SEAT:
            self.status.configure(text=f"자리 이탈: 얼굴이 보이지 않습니다 ({out_ms / 1000:.1f}초)", bg=RED_BG, fg=RED_TEXT)
        elif active is DepartureKind.GAZE:
            self.status.configure(text=f"동공 이탈: 화면을 봐 주세요 ({out_ms / 1000:.1f}초)", bg=RED_BG, fg=RED_TEXT)
        else:
            self.status.configure(text="정상: 화면을 보고 있습니다", bg=OK_BG, fg=OK)
        self.fps_label.configure(text=f"처리 속도 {self.fps:.1f} FPS")
        self._show_frame(frame, result)
        if remaining <= 0:
            self._end_exam(now_ms)

    def _show_frame(self, frame, result) -> None:
        view = frame.copy()
        height, width = view.shape[:2]
        if result.face_landmarks:
            face = result.face_landmarks[0]
            for index in IRIS_CENTER_INDICES:
                if index < len(face):
                    point = (round(face[index].x * width), round(face[index].y * height))
                    cv2.circle(view, point, 3, (255, 0, 255), -1, cv2.LINE_AA)
        view = cv2.flip(view, 1)  # 화면 표시만 거울처럼 반전 (판정은 원본 기준)
        if width != VIDEO_WIDTH:
            view = cv2.resize(view, (VIDEO_WIDTH, round(height * VIDEO_WIDTH / width)))
        image = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image)
        self.video.configure(image=self.photo, width=image.width, height=image.height)

    # ---------- 프레임 처리 (보정·시험 공통) ----------
    def _tick(self) -> None:
        if self.mode not in ("calibration", "exam") or self.camera is None:
            return
        try:
            frame, now_ms = self.camera.read()
            result = self.landmarks.detect(frame, now_ms)
        except (RuntimeError, ValueError, cv2.error) as error:
            if self.mode == "exam":
                self._end_exam(self._last_tick_ms)
                messagebox.showerror(
                    "카메라 오류", f"{error}\n지금까지의 기록을 저장하고 시험을 종료했습니다.",
                    parent=self.root,
                )
            else:
                self._fail(str(error))
            return
        if self._last_tick_ms is not None and now_ms > self._last_tick_ms:
            current = 1000 / (now_ms - self._last_tick_ms)
            self.fps = current if self.fps == 0 else self.fps * 0.9 + current * 0.1
        self._last_tick_ms = now_ms

        if self.mode == "calibration":
            sample = pupil_sample(result, now_ms, frame.shape[1], frame.shape[0])
            self.calib_face = sample.face_detected
            self.calibrator.add(sample)
            if self.calibrator.done:
                self._finish_calibration(now_ms)
            else:
                self._draw_calibration()
        else:
            self._update_exam(frame, result, now_ms)
        if self.mode in ("calibration", "exam"):
            self.root.after(1, self._tick)

    # ---------- 화면 3: 결과 조회 ----------
    def show_result_auth(self, directory) -> None:
        self._clear()
        self.mode = "result"
        f = self.font
        sessions = list_sessions()
        wrap = tk.Frame(self.container, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")
        title = "시험이 끝났습니다" if directory is not None else "지난 기록 조회"
        tk.Label(wrap, text=title, font=(f, 26, "bold"), bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(
            wrap,
            text="결과를 보려면 감독관 패스워드로 본인 확인이 필요합니다.",
            font=(f, 13),
            bg=BG,
            fg=MUTED,
        ).pack(anchor="w", pady=(6, 18))
        card = tk.Frame(wrap, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        card.pack(fill="x")
        body = tk.Frame(card, bg=PANEL)
        body.pack(padx=32, pady=(14, 28), fill="x")

        tk.Label(body, text="기록", font=(f, 12, "bold"), bg=PANEL, fg=TEXT).pack(
            anchor="w", pady=(16, 6)
        )
        names = [p.name for p in sessions]
        self.session_choice = ttk.Combobox(
            body, values=names, state="readonly", font=(f, 13), width=40
        )
        if directory is not None and directory.name in names:
            self.session_choice.set(directory.name)
        elif names:
            self.session_choice.set(names[0])
        self.session_choice.pack(fill="x")
        self._sessions = {p.name: p for p in sessions}
        self.auth_pw = self._field(body, "감독관 패스워드", show="●")
        if directory is not None and self.final_chain:
            tk.Label(
                body,
                text=f"최종 체인값 (따로 보관하면 기록 바꿔치기를 확인할 수 있습니다)\n{self.final_chain}",
                font=(f, 10),
                bg=PANEL,
                fg=MUTED,
                justify="left",
                wraplength=520,
            ).pack(anchor="w", pady=(14, 0))
        self.auth_error = tk.Label(body, text="" if names else "저장된 기록이 없습니다.", font=(f, 11, "bold"), bg=PANEL, fg=RED_TEXT)
        self.auth_error.pack(anchor="w", pady=(8, 0))

        buttons = tk.Frame(wrap, bg=BG)
        buttons.pack(fill="x", pady=(18, 0))
        self._button(buttons, "결과 조회", self._open_result).pack(side="right")
        self._button(buttons, "처음으로", self.show_setup, primary=False).pack(side="left")
        self.auth_pw.focus_set()
        self.auth_pw.bind("<Return>", lambda _e: self._open_result())

    def _open_result(self) -> None:
        directory = self._sessions.get(self.session_choice.get())
        if directory is None:
            self.auth_error.configure(text="기록을 선택하세요.")
            return
        try:
            session = open_session(directory, self.auth_pw.get())
        except PasswordError as error:
            self.auth_error.configure(text=str(error))
            return
        except (OSError, ValueError, KeyError) as error:
            self.auth_error.configure(text=f"기록을 열 수 없습니다: {error}")
            return
        self.show_result(session)

    def show_result(self, session) -> None:
        self._clear()
        f = self.font
        header = session.header
        items = build_report(session)
        page = tk.Frame(self.container, bg=BG)
        page.pack(fill="both", expand=True, padx=32, pady=24)

        top = tk.Frame(page, bg=BG)
        top.pack(fill="x")
        tk.Label(top, text="시험 결과: 의심 정황", font=(f, 24, "bold"), bg=BG, fg=TEXT).pack(side="left")
        self._button(top, "처음으로", self.show_setup, primary=False).pack(side="right")
        self._button(
            top, "다른 기록 조회", lambda: self.show_result_auth(None), primary=False
        ).pack(side="right", padx=(0, 8))
        started = header["started_at"].replace("T", " ")[:19]
        tk.Label(
            page,
            text=f"{header['student']}  ·  시작 {started}  ·  시험 시간 {format_clock(header['exam_seconds'])}",
            font=(f, 12),
            bg=BG,
            fg=MUTED,
        ).pack(anchor="w", pady=(6, 12))

        if session.intact:
            banner_text = "무결성 정상: 모든 체인값이 일치합니다. 기록이 바뀌지 않았습니다."
            banner_bg, banner_fg = OK_BG, OK
        else:
            banner_text = "무결성 문제 발견\n" + "\n".join(f"· {p}" for p in session.problems)
            banner_bg, banner_fg = RED_BG, RED_TEXT
        tk.Label(
            page, text=banner_text, font=(f, 12, "bold"), bg=banner_bg, fg=banner_fg,
            justify="left", anchor="w", padx=16, pady=12,
        ).pack(fill="x")

        counts = {kind: [0, 0] for kind in DepartureKind}
        for event in session.events:
            counts[event.kind][0] += 1
            counts[event.kind][1] += event.duration_ms
        summaries = {k: KindSummary(c, t, session.criteria) for k, (c, t) in counts.items()}
        suspicious = any(s.suspicious for s in summaries.values())
        stats = tk.Frame(page, bg=BG)
        stats.pack(fill="x", pady=(12, 4))
        for kind, label in ((DepartureKind.GAZE, "동공 이탈"), (DepartureKind.SEAT, "자리 이탈")):
            s = summaries[kind]
            tk.Label(
                stats, text=f"{label}  {s.count}회 · {s.total_ms / 1000:.1f}초",
                font=(f, 13, "bold"), bg=BG, fg=TEXT,
            ).pack(side="left", padx=(0, 28))
        tk.Label(
            stats,
            text="판정: 부정행위 의심 (감독관 확인 필요)" if suspicious else "판정: 의심 기준 미달",
            font=(f, 13, "bold"), bg=BG, fg=RED_TEXT if suspicious else OK,
        ).pack(side="left")
        c = session.criteria
        tk.Label(
            page,
            text=(
                f"기준 (기록 당시): {c.departure_ms / 1000:.0f}초 넘게 벗어나면 이탈 1회 · "
                f"종류별 {c.suspicious_count}회 또는 누적 {c.suspicious_total_ms / 1000:.0f}초 이상이면 의심    "
                "빨강 = 이탈 1건   노랑 = 누적 기준 도달"
            ),
            font=(f, 10), bg=BG, fg=MUTED,
        ).pack(anchor="w", pady=(0, 10))

        list_box = tk.Frame(page, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        list_box.pack(fill="both", expand=True)
        canvas = tk.Canvas(list_box, bg=PANEL, highlightthickness=0)
        scroll = ttk.Scrollbar(list_box, orient="vertical", command=canvas.yview)
        rows = tk.Frame(canvas, bg=PANEL)
        rows.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.winfo_exists() and canvas.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units"),
        )

        if not items:
            tk.Label(rows, text="의심 정황이 없습니다.", font=(f, 14), bg=PANEL, fg=MUTED).pack(pady=40)
        for item in items:
            self._result_row(rows, item, header["student"])

    def _result_row(self, parent, item, student) -> None:
        f = self.font
        red = item.color == RED
        bg, fg = (RED_BG, RED_TEXT) if red else (YELLOW_BG, YELLOW_TEXT)
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", padx=12, pady=(12, 0))
        chip = tk.Label(row, text="이탈" if red else "누적", font=(f, 11, "bold"), bg=fg, fg="#FFFFFF", padx=10, pady=4)
        chip.pack(side="left", padx=(16, 14), pady=16)
        text = tk.Frame(row, bg=bg)
        text.pack(side="left", fill="x", expand=True, pady=12)
        tk.Label(text, text=item.title, font=(f, 13, "bold"), bg=bg, fg=TEXT, anchor="w").pack(anchor="w")
        detail = item.detail + ("" if item.valid else "   ※ 체인값 불일치: 이 기록은 바뀌었을 수 있습니다")
        tk.Label(text, text=detail, font=(f, 12), bg=bg, fg=fg, anchor="w").pack(anchor="w", pady=(2, 0))
        tk.Button(
            row, text="체인값  ›", font=(f, 11, "bold"), bg=PANEL, fg=TEXT, relief="flat",
            activebackground=TRACK, bd=0, padx=16, pady=8, cursor="hand2",
            highlightthickness=1, highlightbackground=LINE,
            command=lambda: self._show_chain(item, student),
        ).pack(side="right", padx=16)

    def _show_chain(self, item, student) -> None:
        f = self.font
        win = tk.Toplevel(self.root)
        win.title("체인값 조회")
        win.configure(bg=PANEL)
        win.transient(self.root)
        win.resizable(False, False)
        body = tk.Frame(win, bg=PANEL)
        body.pack(padx=28, pady=24)
        tk.Label(body, text=f"{item.seq}번 기록의 체인값", font=(f, 15, "bold"), bg=PANEL, fg=TEXT).pack(anchor="w")
        tk.Label(body, text=f"{item.title}\n{item.detail}", font=(f, 11), bg=PANEL, fg=MUTED, justify="left").pack(anchor="w", pady=(6, 12))
        value = tk.Entry(body, font=("Consolas", 12), width=66, relief="flat", bg="#FAFBFC", fg=TEXT, highlightthickness=1, highlightbackground=LINE)
        value.insert(0, item.chain)
        value.configure(state="readonly", readonlybackground="#FAFBFC")
        value.pack(fill="x", ipady=6)
        state = "일치: 다시 계산한 값과 같습니다." if item.valid else "불일치: 다시 계산한 값과 다릅니다. 기록이 바뀌었을 수 있습니다."
        tk.Label(body, text=state, font=(f, 11, "bold"), bg=PANEL, fg=OK if item.valid else RED_TEXT).pack(anchor="w", pady=(10, 0))
        tk.Label(
            body,
            text="체인값 = HMAC-SHA256(감독관 패스워드로 만든 키, 이전 체인값 + 이 기록).\n"
            "기록 한 줄만 고치거나 지워도 그 뒤의 값이 맞지 않게 됩니다.",
            font=(f, 10), bg=PANEL, fg=MUTED, justify="left",
        ).pack(anchor="w", pady=(8, 14))

        def copy():
            win.clipboard_clear()
            win.clipboard_append(item.chain)
            copy_button.configure(text="복사됨")

        buttons = tk.Frame(body, bg=PANEL)
        buttons.pack(fill="x")
        self._button(buttons, "닫기", win.destroy).pack(side="right")
        copy_button = self._button(buttons, "복사", copy, primary=False)
        copy_button.pack(side="right", padx=(0, 8))


def run(model_path) -> int:
    """화면을 띄우고 창이 닫힐 때까지 실행한다."""
    from src.landmarks import FaceLandmarkDetector

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    try:
        detector = FaceLandmarkDetector(model_path)
    except (OSError, RuntimeError, ValueError) as error:
        root.withdraw()
        messagebox.showerror("모델 오류", str(error))
        root.destroy()
        print(f"실행 실패: {error}", file=sys.stderr)
        return 1
    with detector:
        app = ExamApp(root, detector)
        try:
            root.mainloop()
        finally:
            app._stop_camera()
    return 0
