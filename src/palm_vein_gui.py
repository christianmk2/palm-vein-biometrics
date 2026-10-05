"""
Palm Vein Biometrics with Temperature Scanner - Contact Tracing Application
----------------------------------------------------------------------------
Tkinter GUI for a Raspberry Pi kiosk that:
  1. Registers a new person (personal details + palm images of both hands).
  2. Verifies a returning person (captures a palm image + temperature,
     runs it through a trained CNN, and logs the temperature reading).

This is a cleaned-up version of the original thesis code. Behavior is kept
as close to the original as possible - see the notes at the bottom of the
accompanying explanation for what was changed and why.
"""

import os
import glob
import uuid
import time
import timeit

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot

import tkinter as tk
from tkinter import (
    Frame, Label, Button, Entry, OptionMenu, StringVar,
    CENTER, LEFT, RIGHT, FLAT, font, messagebox,
)

import mysql.connector

import board
import busio
import adafruit_mlx90614

from tensorflow import keras


# --------------------------------------------------------------------------
# Configuration - EDIT THESE FOR YOUR SETUP
# --------------------------------------------------------------------------
MAIN_FOLDER = "/home/pi/CapturedImages/"          # raw captured hand photos
PROCESSED_FOLDER = "/home/pi/Preprocess_Pasig/"   # preprocessed training images
TEMP_FOLDER = "/home/pi/temp/"                    # scratch space for verification
MODEL_PATH = "/home/pi/TESTING H5/8userjay.h5"    # trained classifier

DB_HOST = "192.168.100.207"
DB_CONFIG = dict(host=DB_HOST, user="root", password="", database="tracking_db")

REG_IMAGE_COUNT = 100          # images captured per hand during registration
ESTABLISHMENT_ID = 10          # TODO: make configurable if you run more than one kiosk

APP_TITLE = (
    "DEVELOPMENT OF PALM VEIN BIOMETRICS \n"
    "WITH TEMPERATURE SCANNER \n"
    "FOR CONTACT TRACING APPLICATION"
)


# --------------------------------------------------------------------------
# Image processing helpers
# (Both registration and verification used the exact same crop -> denoise ->
#  CLAHE -> invert -> skeletonize -> threshold -> crop pipeline. It was
#  duplicated in full in the original file; it's a single function here.)
# --------------------------------------------------------------------------
def _skeletonize(gray_image):
    img = gray_image.copy()
    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        delta = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, delta)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def extract_palm_features(bgr_image, roi=(273, 94, 214, 195), crop=(4, 4, 205, 187)):
    """Runs the same crop/denoise/CLAHE/skeletonize pipeline used for both
    registration images and verification images."""
    x, y, w, h = roi
    region = bgr_image[y:y + h, x:x + w]

    denoised = cv2.fastNlMeansDenoising(region)

    kernel = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    yuv = cv2.cvtColor(opened, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    inverted = cv2.bitwise_not(equalized)

    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    skeleton = _skeletonize(gray)
    _, binary = cv2.threshold(skeleton, 5, 255, cv2.THRESH_BINARY)

    x1, y1, w1, h1 = crop
    return binary[y1:y1 + h1, x1:x1 + w1]


def to_rgb(image):
    """The classifier expects a 3-channel RGB image. NOTE: the original code
    always called cv2.cvtColor(img, COLOR_BGR2RGB) here, but by this point
    the image is single-channel grayscale (see extract_palm_features above),
    which raises a cv2 assertion error. This guards against that - see the
    write-up for details."""
    if image.ndim == 2 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def add_temperature_offset(raw_temp, offset=1.0):
    """Applies the same fixed calibration offset the original code used."""
    return round(float(raw_temp), 2) + offset


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def insert_person(data):
    """Inserts a newly registered person. Raises mysql.connector.Error on
    failure (and shows a messagebox) so the caller can decide whether to
    continue to image capture."""
    query = (
        "INSERT INTO persons "
        "(lastname, firstname, middlename, gender, contact_number, "
        " email, address, province, age, tracking_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    values = (
        data["lastname"], data["firstname"], data["middlename"], data["gender"],
        data["contact"], data["email"], data["address"], data["province"],
        data["age"], data["tracking_id"],
    )

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()
        messagebox.showerror("Database Error", str(err))
        raise
    finally:
        conn.close()


def record_temperature_reading(tracking_id, temperature):
    """Logs a temperature reading against an existing person and returns
    their latest (name, temperature, timestamp) record, or None if nothing
    matched. Uses parameterized queries throughout (the original built the
    tracking-id clause with raw string concatenation, which is a SQL
    injection risk)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO temporary (establishment_id, temperature, tracking_id) "
            "VALUES (%s, %s, %s)",
            (ESTABLISHMENT_ID, temperature, tracking_id),
        )
        conn.commit()

        cursor.execute(
            "INSERT INTO person_tracks (person_id, establishment_id, temperature, tracking_id) "
            "SELECT persons.id, temporary.establishment_id, temporary.temperature, temporary.tracking_id "
            "FROM persons JOIN temporary ON persons.tracking_id = temporary.tracking_id "
            "WHERE temporary.tracking_id = %s",
            (tracking_id,),
        )
        conn.commit()

        cursor.execute(
            "DELETE FROM temporary WHERE tracking_id = %s",
            (tracking_id,),
        )
        conn.commit()

        cursor.execute(
            "SELECT persons.firstname, persons.middlename, persons.lastname, "
            "       person_tracks.temperature, person_tracks.date_created "
            "FROM persons JOIN person_tracks "
            "ON persons.tracking_id = person_tracks.tracking_id "
            "WHERE persons.tracking_id = %s "
            "ORDER BY person_tracks.date_created DESC LIMIT 1",
            (tracking_id,),
        )
        return cursor.fetchone()
    except mysql.connector.Error as err:
        conn.rollback()
        messagebox.showerror("Database Error", str(err))
        return None
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class PalmVeinApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Home")
        self.root.geometry("1250x720")
        self.root.resizable(False, False)
        self.root.configure(background="white")

        self.font_button = font.Font(family="Helvetica", size=36, weight="bold")

        i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
        self.temp_sensor = adafruit_mlx90614.MLX90614(i2c)

        self.current_frame = None          # the single "page" frame on screen
        self.registration_image_counter = 0
        self.verify_start_time = None

        self.show_home_screen()

    # ---- screen management -------------------------------------------------
    def _clear_screen(self):
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = None

    def _new_page(self):
        """A full-window container frame. Everything for a screen lives
        inside it, so switching screens is a single destroy() + rebuild
        instead of individually place_forget()-ing a dozen widgets."""
        self._clear_screen()
        page = Frame(self.root)
        page.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.current_frame = page
        return page

    # ---- home screen --------------------------------------------------------
    def show_home_screen(self):
        page = self._new_page()
        self.root.title("Home")

        Label(
            page, text=APP_TITLE, width=50, height=4,
            bg="white", fg="black", anchor=CENTER, font=("bold", 32),
        ).place(anchor=CENTER, relx=0.5, rely=0.12)

        button_row = Frame(page, height=300, width=450, bg="skyblue", bd=1)
        button_row.place(anchor=CENTER, relx=0.5, rely=0.50)

        Button(
            button_row, text="REGISTER\nUSER", font=self.font_button, width=10,
            bg="dodgerblue4", fg="white", activebackground="steelblue",
            activeforeground="white", command=self.show_register_screen,
        ).pack(anchor=CENTER, side=LEFT, pady=100, padx=(150, 50))

        Button(
            button_row, text="VERIFY\nUSER", font=self.font_button, width=10,
            bg="dodgerblue4", fg="white", activebackground="steelblue",
            activeforeground="white", command=self.show_verify_screen,
        ).pack(anchor=CENTER, side=LEFT, pady=100, padx=(25, 19))

        Button(
            button_row, text="EXIT\nPROGRAM", font=self.font_button, width=10,
            bg="dodgerblue4", fg="white", activebackground="steelblue",
            activeforeground="white", command=self.root.destroy,
        ).pack(anchor=CENTER, side=RIGHT, pady=100, padx=(50, 150))

    # ---- registration: form --------------------------------------------------
    def show_register_screen(self):
        page = self._new_page()
        self.root.title("Register")

        Label(
            page, text="USER REGISTRATION FORM", width=32, height=1,
            bg="dodgerblue2", fg="white", anchor=CENTER, font=("bold", 32),
        ).place(anchor=CENTER, relx=0.5, rely=0.05)

        form = Frame(page, height=600, width=800, bg="skyblue", bd=1, relief=FLAT)
        form.place(anchor=CENTER, relx=0.5, rely=0.52)

        entries = {}

        Label(form, text="Last Name", width=10, font=("bold", 15)).place(anchor=CENTER, relx=0.20, rely=0.12)
        entries["lastname"] = Entry(form, width=18, font=("bold", 20))
        entries["lastname"].place(anchor=CENTER, relx=0.20, rely=0.05)

        Label(form, text="First Name", width=10, font=("bold", 15)).place(anchor=CENTER, relx=0.60, rely=0.12)
        entries["firstname"] = Entry(form, width=18, font=("bold", 20))
        entries["firstname"].place(anchor=CENTER, relx=0.60, rely=0.05)

        Label(form, text="M.I.", width=4, font=("bold", 15)).place(anchor=CENTER, relx=0.88, rely=0.12)
        entries["middlename"] = Entry(form, width=4, font=("bold", 20))
        entries["middlename"].place(anchor=CENTER, relx=0.88, rely=0.05)

        Label(form, text="Age", width=4, font=("bold", 15)).place(anchor=CENTER, relx=0.15, rely=0.27)
        entries["age"] = Entry(form, width=4, font=("bold", 20))
        entries["age"].place(anchor=CENTER, relx=0.15, rely=0.20)

        Label(form, text="Gender", width=10, font=("bold", 15)).place(anchor=CENTER, relx=0.37, rely=0.27)
        gender_var = StringVar(value="MALE")
        OptionMenu(form, gender_var, "MALE", "FEMALE").place(anchor=CENTER, relx=0.42, rely=0.27)

        Label(form, text="Contact", width=10, font=("bold", 15)).place(anchor=CENTER, relx=0.75, rely=0.27)
        entries["contact"] = Entry(form, width=15, font=("bold", 20))
        entries["contact"].place(anchor=CENTER, relx=0.75, rely=0.20)

        Label(form, text="Email", width=10, font=("bold", 15)).place(anchor=CENTER, relx=0.50, rely=0.45)
        entries["email"] = Entry(form, width=35, font=("bold", 20))
        entries["email"].place(anchor=CENTER, relx=0.50, rely=0.37)

        Label(form, text="Address", width=10, font=("bold", 15)).place(anchor=CENTER, relx=0.50, rely=0.62)
        entries["address"] = Entry(form, width=30, font=("bold", 20))
        entries["address"].place(anchor=CENTER, relx=0.50, rely=0.55)

        Label(form, text="Province", width=10, font=("bold", 15)).place(anchor=CENTER, relx=0.50, rely=0.79)
        entries["province"] = Entry(form, width=10, font=("bold", 20))
        entries["province"].place(anchor=CENTER, relx=0.50, rely=0.72)

        Button(
            form, text="BACK", font=("bold", 20), width=10, bg="dodgerblue4", fg="white",
            activebackground="steelblue", activeforeground="white",
            command=self.show_home_screen,
        ).place(anchor=CENTER, relx=0.35, rely=0.90)

        Button(
            form, text="NEXT", font=("bold", 20), width=10, bg="dodgerblue4", fg="white",
            activebackground="steelblue", activeforeground="white",
            command=lambda: self.handle_register_submit(entries, gender_var),
        ).place(anchor=CENTER, relx=0.65, rely=0.90)

    def handle_register_submit(self, entries, gender_var):
        tracking_id = str(uuid.uuid4().int)[:5]
        data = {
            "lastname": entries["lastname"].get(),
            "firstname": entries["firstname"].get(),
            "middlename": entries["middlename"].get(),
            "gender": gender_var.get(),
            "contact": entries["contact"].get(),
            "email": entries["email"].get(),
            "address": entries["address"].get(),
            "province": entries["province"].get(),
            "age": entries["age"].get(),
            "tracking_id": tracking_id,
        }

        try:
            insert_person(data)
        except mysql.connector.Error:
            return  # error already shown to the user; stay on the form

        messagebox.showinfo("Information", "User inserted successfully...")
        self.root.after(1000, lambda: self.start_registration_capture(tracking_id))

    # ---- registration: image capture -----------------------------------------
    def start_registration_capture(self, tracking_id):
        left_dir = os.path.join(MAIN_FOLDER, str(tracking_id), "left")
        right_dir = os.path.join(MAIN_FOLDER, str(tracking_id), "right")
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)

        self._clear_screen()
        messagebox.showinfo("Steps", "Please place your left hand and press SPACE to start capturing")
        self.root.after(1000, lambda: self.capture_hand(
            "left", tracking_id, lambda: self._after_left_capture(tracking_id)))

    def _after_left_capture(self, tracking_id):
        messagebox.showinfo(
            "Steps", "Capture finished. Please place your right hand and press SPACE to start capturing")
        self.root.after(1000, lambda: self.capture_hand(
            "right", tracking_id, lambda: self._after_right_capture(tracking_id)))

    def _after_right_capture(self, tracking_id):
        messagebox.showinfo("Steps", "Capture finished. Images are now being preprocessed.")
        self.show_preprocessing_screen(tracking_id)

    def capture_hand(self, hand, tracking_id, on_done):
        """Opens the camera, waits for SPACE to begin, then captures
        REG_IMAGE_COUNT grayscale frames into MAIN_FOLDER/<id>/<hand>/."""
        self.root.title(f"Capturing {hand.title()}")
        window_name = f"Capture {hand.title()}-Hand"
        save_dir = os.path.join(MAIN_FOLDER, str(tracking_id), hand)

        cam = cv2.VideoCapture(0)

        while True:
            _, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(window_name, gray)
            key = cv2.waitKey(1)
            if key % 256 == 32:  # SPACE
                break

        for i in range(REG_IMAGE_COUNT):
            # A few throwaway reads keep the preview window responsive
            # between saved captures (kept from the original).
            for _ in range(3):
                _, preview = cam.read()
                cv2.imshow(window_name, preview)
                cv2.waitKey(1)

            _, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(window_name, gray)
            cv2.waitKey(1)
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), gray)
            print(i)

        cam.release()
        cv2.destroyAllWindows()
        on_done()

    # ---- registration: preprocessing -----------------------------------------
    def show_preprocessing_screen(self, tracking_id):
        page = self._new_page()
        self.root.title("Preprocessing")

        frame = Frame(page, height=300, width=450, bg="skyblue", bd=1)
        frame.place(anchor=CENTER, relx=0.5, rely=0.50)
        Label(
            frame, text="Preprocessing Images\nPlease Wait...",
            fg="black", bg="skyblue", font=("bold", 80),
        ).pack(pady=10, anchor=CENTER)

        self.root.after(1000, lambda: self.run_preprocessing(tracking_id))

    def run_preprocessing(self, tracking_id):
        user_folder = os.path.join(MAIN_FOLDER, str(tracking_id)) + os.sep
        image_paths = glob.glob(user_folder + "*/*.jpg")

        for image_path in image_paths:
            raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            processed = extract_palm_features(raw_image)
            self._save_processed_image(processed, tracking_id)

        self.show_home_screen()

    def _save_processed_image(self, image, tracking_id):
        """NOTE: kept exactly as the original numbered things - the first
        100 processed images go into the "...R" folder and the next 100
        into the "...L" folder. That looks backwards, but it's the
        convention the existing trained model (MODEL_PATH) expects, so
        changing it would break compatibility with that model."""
        left_dir = f"{PROCESSED_FOLDER}{tracking_id}L"
        right_dir = f"{PROCESSED_FOLDER}{tracking_id}R"
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)

        self.registration_image_counter += 1
        ctr = self.registration_image_counter

        if ctr <= 100:
            pyplot.imsave(f"{right_dir}/{ctr}.jpg", image, cmap="gray")
        elif ctr <= 200:
            pyplot.imsave(f"{left_dir}/{ctr}.jpg", image, cmap="gray")

        if ctr == 200:
            self.registration_image_counter = 0

    # ---- verification --------------------------------------------------------
    def show_verify_screen(self):
        page = self._new_page()
        self.root.title("Verify")

        frame = Frame(page, height=100, width=100, bg="skyblue", bd=1)
        frame.place(anchor=CENTER, relx=0.5, rely=0.50)

        Button(
            frame, text="Take\nPalm Photo", font=self.font_button, height=2, width=10,
            bg="dodgerblue4", fg="white", activebackground="steelblue",
            activeforeground="white", command=self.capture_verification_photo,
        ).pack(pady=(100, 0), padx=(100, 100), anchor=CENTER)

        Button(
            frame, text="Back", font=self.font_button, height=2, width=10,
            bg="dodgerblue4", fg="white", activebackground="steelblue",
            activeforeground="white", command=self.show_home_screen,
        ).pack(pady=(20, 100), padx=(100, 100), anchor=CENTER)

    def capture_verification_photo(self):
        self._clear_screen()
        self.root.title("Capturing")
        self.verify_start_time = timeit.default_timer()

        cam = cv2.VideoCapture(0)
        capture_started = time.time()
        target_temp = None

        while True:
            raw_temp = "{:.1f}".format(self.temp_sensor.object_temperature)
            target_temp = add_temperature_offset(raw_temp)

            _, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Verify Image", gray)
            print("Target Temperature:", target_temp, "\u00b0C")

            cv2.waitKey(1)
            if time.time() - capture_started >= 4:
                break

        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Verify Image", gray)
        cv2.waitKey(1)

        capture_path = os.path.join(TEMP_FOLDER, "test.jpg")
        cv2.imwrite(capture_path, gray)

        cam.release()
        cv2.destroyAllWindows()

        self.process_verification(capture_path, target_temp)

    def process_verification(self, capture_path, target_temp):
        page = self._new_page()
        Label(page, text="Waiting").place(anchor=CENTER, relx=0.5, rely=0.5)

        raw_image = cv2.imread(capture_path, cv2.IMREAD_COLOR)
        processed = extract_palm_features(raw_image)

        self.root.after(500, lambda: self.run_model_prediction(processed, target_temp))

    def run_model_prediction(self, processed_image, target_temp):
        model = keras.models.load_model(MODEL_PATH)
        pyplot.imsave(os.path.join(TEMP_FOLDER, "test.jpg"), processed_image, cmap="gray")

        rgb_image = to_rgb(processed_image)
        resized = Image.fromarray(rgb_image, "RGB").resize((224, 224))
        input_batch = np.array(resized).reshape(1, 224, 224, 3).astype("float32") / 255.0

        predictions = model.predict(input_batch)
        predicted_class = int(predictions.argmax())
        confidence = float(predictions.flatten()[predicted_class])

        print(f"Confidence: {confidence}")
        print(f"Predicted class: {predicted_class}")

        if confidence < 0.80:
            self.show_no_match_screen()
            return

        # Training categories are sorted "<id>L" / "<id>R" pairs, two per
        # registered user, so every user occupies two consecutive class
        # indices (this replaces the original's odd/even counting loop,
        # which works out to the same thing: predicted_class // 2).
        user_index = predicted_class // 2
        tracking_ids = sorted(os.listdir(MAIN_FOLDER))

        if user_index >= len(tracking_ids):
            self.show_no_match_screen()
            return

        tracking_id = tracking_ids[user_index]
        result = record_temperature_reading(tracking_id, target_temp)

        if result is None:
            self.show_no_match_screen()
            return

        elapsed = timeit.default_timer() - (self.verify_start_time or timeit.default_timer())
        print("Time:", elapsed)

        self.show_match_result_screen(result)

    def show_match_result_screen(self, result):
        firstname, middlename, lastname, temperature, date_created = result

        page = self._new_page()
        frame = Frame(page, height=300, width=450, bg="skyblue", bd=1)
        frame.place(anchor=CENTER, relx=0.5, rely=0.50)

        Label(frame, bg="skyblue", fg="black", font=("bold", 24),
              text="Successfully Verified\nWelcome").pack(pady=10, anchor=CENTER)
        Label(frame, bg="skyblue", fg="black", font=("bold", 16),
              text=f"Name: {firstname} {middlename} {lastname}").pack(pady=10, anchor=CENTER)
        Label(frame, bg="skyblue", fg="black", font=("bold", 16),
              text=f"Temperature: {temperature}").pack(pady=10, anchor=CENTER)
        Label(frame, bg="skyblue", fg="black", font=("bold", 16),
              text=f"Date/Time: {date_created}").pack(pady=10, anchor=CENTER)

        self.root.after(5000, self.show_home_screen)

    def show_no_match_screen(self):
        page = self._new_page()
        frame = Frame(page, height=300, width=450, bg="skyblue", bd=1)
        frame.place(anchor=CENTER, relx=0.5, rely=0.50)

        Label(frame, bg="skyblue", fg="black", font=("bold", 24),
              text="No existing matches found\nPlease try again later").pack(pady=10, anchor=CENTER)

        self.root.after(5000, self.show_home_screen)


def main():
    root = tk.Tk()
    PalmVeinApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
