from random import randint
from turtle import title
import uuid
import pandas as pd
import csv
import os
import cv2
import tkinter as tk
from tkinter import font
from tkinter import *
from tkcalendar import DateEntry
from datetime import datetime
import mysql.connector
from tkinter import ttk, messagebox
import numpy as np
#from picamera import PiCamera
from time import sleep
import time
from matplotlib import pyplot
from PIL import ImageTk, Image
import board
import busio as io
import adafruit_mlx90614
from tensorflow import keras
import glob
import timeit
#import h5py

#EDIT ALL DIRECTORY
main_folder = '/home/pi/CapturedImages/'
processed_path2 = '/home/pi/Preprocess_Pasig/'
temp_folder = '/home/pi/temp/'
#h5_model = '/home/pi/TESTING H5/finalmodel.h5' (3 user)
h5_model = '/home/pi/TESTING H5/8userjay.h5'
host_id = "192.168.100.207"
reg_img_count = 100
ctr = 0
start, end = 0,0

def exit():
    window.destroy()

##########3#value add###############3
def Add(entry_last,entry_first,entry_middle,var,sex,entry_contact,entry_email,entry_address,entry_province,entry_zipcode,krazynum,
        label_0,lastname,firstname,middlename,droplist,contact,email,address,province,zipcode,addbutton,backadd,frame4):
    lname = entry_last.get()
    fname = entry_first.get()
    mname = entry_middle.get()
    seggs = var.get()
    trackid = krazynum.get()
    contno = entry_contact.get()
    mail = entry_email.get()
    addr = entry_address.get()
    prov = entry_province.get()
    zcode = entry_zipcode.get()

    mysqldb=mysql.connector.connect(host=host_id,user="root",password="",database="tracking_db")
    mycursor=mysqldb.cursor()

    try:
        sql = "INSERT INTO  persons (lastname,firstname,middlename,gender,contact_number,email,address,province,age,tracking_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (lname,fname,mname,seggs,contno,mail,addr,prov,zcode,trackid)
        mycursor.execute(sql, val)
        mysqldb.commit()
        lastid = mycursor.lastrowid
        messagebox.showinfo("information", "User inserted successfully...")
        entry_last.delete(0, END)
        entry_first.delete(0, END)
        entry_middle.delete(0, END)
        var.set("MALE")
        entry_contact.delete(0, END)
        entry_email.delete(0, END)
        entry_address.delete(0, END)
        entry_province.delete(0, END)
        entry_zipcode.delete(0, END)
        entry_last.focus_set()
        window.after(1000, lambda:leftreg_image_capture(trackid, entry_last,entry_first,entry_middle,var,sex,entry_contact,entry_email,entry_address,entry_province,entry_zipcode,
            label_0,lastname,firstname,middlename,droplist,contact,email,address,province,zipcode,addbutton,backadd,frame4))
    except Exception as e:
        print(e)
        mysqldb.rollback()
        mysqldb.close()
    # NOTE: mysqldb is never closed on the success path - only in the except
    # branch above. That's how the original had it.

def leftreg_image_capture(trackid, entry_last,entry_first,entry_middle,var,sex,entry_contact,entry_email,entry_address,entry_province,entry_zipcode,
                           label_0,lastname,firstname,middlename,droplist,contact,email,address,province,zipcode,addbutton,backadd,frame4):
    entry_last.place_forget()
    entry_first.place_forget()
    entry_middle.place_forget()
    sex.place_forget()
    entry_contact.place_forget()
    entry_email.place_forget()
    entry_address.place_forget()
    entry_province.place_forget()
    entry_zipcode.place_forget()
    label_0.place_forget()
    lastname.place_forget()
    firstname.place_forget()
    middlename.place_forget()
    droplist.place_forget()
    contact.place_forget()
    email.place_forget()
    address.place_forget()
    province.place_forget()
    zipcode.place_forget()
    addbutton.place_forget()
    backadd.place_forget()
    frame4.place_forget()

    msgbox = messagebox.showinfo("Steps", "Please place your left hand and press start to start capturing")

    left_folder = main_folder + str(trackid) + "/left/"
    right_folder = main_folder + str(trackid) + "/right/"
    os.makedirs(left_folder)
    os.makedirs(right_folder)
    window.after(1000, lambda:left_cap(trackid))

def left_cap(trackid):
    window.title("Capturing Left")
    cam = cv2.VideoCapture(0)
    while(True):
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture Left-Hand", gray)
        k = cv2.waitKey(1)
        if k%256 == 32: # SPACE pressed
            break

    for i in range(reg_img_count):
        for j in range(3):
            _, frame2 = cam.read()
            cv2.imshow("Capture Left-Hand", frame2)
            cv2.waitKey(1)
        _, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture Left-Hand", gray)
        k = cv2.waitKey(1)
        cv2.imwrite(main_folder + str(trackid) + "/left/" + str(i) +".jpg", gray)
        print(i)

    cam.release()
    cv2.destroyAllWindows()
    msgbox = messagebox.showinfo("Steps", "Capture Finished, Please place your right hand and press start to start capturing")
    window.after(1000, lambda:right_cap(trackid))

def right_cap(trackid):
    window.title("Capturing Right")
    cam = cv2.VideoCapture(0)
    while(True):
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture Right-Hand", gray)
        k = cv2.waitKey(1)
        if k%256 == 32: # SPACE pressed
            break

    for i in range(reg_img_count):
        for j in range(3):
            _, frame2 = cam.read()
            cv2.imshow("Capture Right-Hand", frame2)
            cv2.waitKey(1)
        _, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture Right-Hand", gray)
        k = cv2.waitKey(1)
        cv2.imwrite(main_folder + str(trackid) + "/right/" + str(i) +".jpg", gray)
        print(i)

    cam.release()
    cv2.destroyAllWindows()
    msgbox = messagebox.showinfo("Steps", "Capture Finished, Images is now being preprocessed")
    window.title("Preprocessing")
    frame7 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame7.place(anchor = CENTER, relx = 0.5, rely = 0.50)
    loadlabel = Label(frame7, text="Preprocessing Images\nPlease Wait...",fg='black',bg= "skyblue",font=("bold", 80))
    loadlabel.pack(pady=(10), anchor=CENTER)
    window.after(1000, lambda:preprocessing_new(trackid,frame7,loadlabel))

################### Preprocessed ########################
def preprocessing_new(trackid,frame7,loadlabel):
    starting = 199
    user = trackid
    user_folder = main_folder + str(user) + "/"
    paths = glob.glob(user_folder + "*/*.jpg")

    x,y,w,h = 273, 94, 214, 195
    users = os.listdir(user_folder)
    users = users[:len(users)]
    print(users)
    for idx, paths in enumerate(paths):
        img_original = cv2.imread(paths, cv2.IMREAD_COLOR)
        roi = img_original[y:y+h,x:x+w]

        # + noise
        noiseReduced = cv2.fastNlMeansDenoising(roi)

        #CLAHE
        kernel = np.ones((7,7),np.uint8)
        img = cv2.morphologyEx(noiseReduced, cv2.MORPH_OPEN, kernel)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        frame = cv2.bitwise_not(img_output)

        #Skeletonize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()
        skel = img.copy()
        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        while cv2.countNonZero(img) > 0:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]
            if cv2.countNonZero(img) == 0:
                break

        ret,frame = cv2.threshold(skel, 5, 255, cv2.THRESH_BINARY)
        x1,y1,w1,h1 = 4, 4, 205, 187
        newframe = frame[y1:y1+h1,x1:x1+w1]
        #folder creation before saving
        folder_creation(idx,user)
        #save image
        saving_image(newframe,user)

    #home
    window.after(1000, lambda:homepage(frame7,loadlabel))

def folder_creation(idx,user):
    l_folder = processed_path2 + str(user) + "L"
    r_folder = processed_path2 + str(user) + "R"
    isExist = os.path.exists(l_folder)
    if isExist == False:
        os.makedirs(l_folder)
        os.makedirs(r_folder)
    elif isExist == True:
        return

def saving_image(image,user):
    global ctr
    ctr += 1
    left_path = (processed_path2 + str(user) + "L")
    right_path = (processed_path2 + str(user) + "R")
    if(ctr<=100):
        pyplot.imsave(f"{right_path}/" + str(ctr) + ".jpg", image, cmap='gray') ##using pyplot to save
    elif(ctr<=200):
        pyplot.imsave(f"{left_path}/" + str(ctr) + ".jpg", image, cmap='gray')
    if (ctr == 200):
        ctr = 0

################### HOME #######################
def homepage(frame7,loadlabel):
    #register
    frame7.place_forget()
    loadlabel.pack_forget()

    frame1 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame1.place(anchor = CENTER, relx = 0.5, rely = 0.50)

    title1 = Label(window, text="DEVELOPMENT OF PALM VEIN BIOMETRICS \nWITH TEMPERATURE SCANNER \n FOR CONTACT TRACING APPLICATION",
                   width=50, height=4, bg='white', fg='black',anchor=CENTER,font=("bold", 32))
    title1.place(anchor = CENTER, relx = 0.5, rely = 0.12)

    #register
    regbutton=Button(frame1, text="REGISTER\n USER",font=myFont, bg="dodgerblue4", fg='white', width=10,
                      activebackground='steelblue',activeforeground='white',
                      command = lambda : register(regbutton, verifybutton, exitbutton,frame1,title1))
    regbutton.pack(anchor = CENTER, side = LEFT, pady=100,padx=(150,50))

    #verify
    verifybutton=Button(frame1,text="VERIFY\n USER", font=myFont,width=10,bg="dodgerblue4", fg='white',
                         activebackground='steelblue',activeforeground='white',
                         command = lambda : verify_real(regbutton, verifybutton, exitbutton, frame1, title1))
    verifybutton.pack(anchor = CENTER, side = LEFT,pady=100,padx=(25,19))

    #exit
    exitbutton=Button(frame1, text="EXIT\n PROGRAM",font=myFont,width=10,bg="dodgerblue4", fg='white',
                       activebackground='steelblue',activeforeground='white', command = exit)
    exitbutton.pack(anchor = CENTER, side = RIGHT,pady=100,padx=(50,150))

def homepage2(takebutton, backbutton, frame5):
    takebutton.pack_forget()
    backbutton.pack_forget()
    frame5.place_forget()

    frame1 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame1.place(anchor = CENTER, relx = 0.5, rely = 0.50)

    title1 = Label(window, text="DEVELOPMENT OF PALM VEIN BIOMETRICS \nWITH TEMPERATURE SCANNER \n FOR CONTACT TRACING APPLICATION",
                   width=50, height=4, bg='white', fg='black',anchor=CENTER,font=("bold", 32))
    title1.place(anchor = CENTER, relx = 0.5, rely = 0.12)

    #register
    regbutton=Button(frame1, text="REGISTER\n USER",font=myFont, bg="dodgerblue4", fg='white',
                      activebackground='steelblue',activeforeground='white',width=10,
                      command = lambda : register(regbutton, verifybutton, exitbutton,frame1,title1))
    regbutton.pack(anchor = CENTER, side = LEFT, pady=100,padx=(150,50))

    #verify
    verifybutton=Button(frame1,text="VERIFY\n USER", font=myFont,width=10,bg="dodgerblue4", fg='white',
                         activebackground='steelblue',activeforeground='white',
                         command = lambda : verify_real(regbutton, verifybutton, exitbutton, frame1, title1))
    verifybutton.pack(anchor = CENTER, side = LEFT,pady=100,padx=(25,19))

    #exit
    exitbutton=Button(frame1, text="EXIT\n PROGRAM",font=myFont,width=10,bg="dodgerblue4", fg='white',
                       activebackground='steelblue',activeforeground='white', command = exit)
    exitbutton.pack(anchor = CENTER, side = RIGHT,pady=100,padx=(50,150))

def homepage3(verlabel, fullnamelabel, templabel, datelabel):
    verlabel.pack_forget()
    templabel.pack_forget()
    fullnamelabel.pack_forget()
    datelabel.pack_forget()

    frame1 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame1.place(anchor = CENTER, relx = 0.5, rely = 0.50)

    title1 = Label(window, text="DEVELOPMENT OF PALM VEIN BIOMETRICS \nWITH TEMPERATURE SCANNER \n FOR CONTACT TRACING APPLICATION",
                   width=50, height=4, bg='white', fg='black',anchor=CENTER,font=("bold", 32))
    title1.place(anchor = CENTER, relx = 0.5, rely = 0.12)

    #register
    regbutton=Button(frame1, text="REGISTER\n USER",font=myFont, bg="dodgerblue4", fg='white',
                      activebackground='steelblue',activeforeground='white',width=10,
                      command = lambda : register(regbutton, verifybutton, exitbutton,frame1,title1))
    regbutton.pack(anchor = CENTER, side = LEFT, pady=100,padx=(150,50))

    #verify
    verifybutton=Button(frame1,text="VERIFY\n USER", font=myFont,width=10,bg="dodgerblue4", fg='white',
                         activebackground='steelblue',activeforeground='white',
                         command = lambda : verify_real(regbutton, verifybutton, exitbutton, frame1, title1))
    verifybutton.pack(anchor = CENTER, side = LEFT,pady=100,padx=(25,19))

    #exit
    exitbutton=Button(frame1, text="EXIT\n PROGRAM",font=myFont,width=10,bg="dodgerblue4", fg='white',
                       activebackground='steelblue',activeforeground='white', command = exit)
    exitbutton.pack(anchor = CENTER, side = RIGHT,pady=100,padx=(50,150))

def homepage4(frame8, verlabel):
    frame8.place_forget()
    verlabel.pack_forget()

    frame1 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame1.place(anchor = CENTER, relx = 0.5, rely = 0.50)

    title1 = Label(window, text="DEVELOPMENT OF PALM VEIN BIOMETRICS \nWITH TEMPERATURE SCANNER \n FOR CONTACT TRACING APPLICATION",
                   width=50, height=4, bg='white', fg='black',anchor=CENTER,font=("bold", 32))
    title1.place(anchor = CENTER, relx = 0.5, rely = 0.12)

    #register
    regbutton=Button(frame1, text="REGISTER\n USER",font=myFont, bg="dodgerblue4", fg='white',
                      activebackground='steelblue',activeforeground='white',width=10,
                      command = lambda : register(regbutton, verifybutton, exitbutton,frame1,title1))
    regbutton.pack(anchor = CENTER, side = LEFT, pady=100,padx=(150,50))

    #verify
    verifybutton=Button(frame1,text="VERIFY\n USER", font=myFont,width=10,bg="dodgerblue4", fg='white',
                         activebackground='steelblue',activeforeground='white',
                         command = lambda : verify_real(regbutton, verifybutton, exitbutton, frame1, title1))
    verifybutton.pack(anchor = CENTER, side = LEFT,pady=100,padx=(25,19))

    #exit
    exitbutton=Button(frame1, text="EXIT\n PROGRAM",font=myFont,width=10,bg="dodgerblue4", fg='white',
                       activebackground='steelblue',activeforeground='white', command = exit)
    exitbutton.pack(anchor = CENTER, side = RIGHT,pady=100,padx=(50,150))

####################verify UI#########################
def verify_real(regbutton, verifybutton, exitbutton,frame1,title1):
    regbutton.pack_forget()
    verifybutton.pack_forget()
    exitbutton.pack_forget()
    frame1.place_forget()
    title1.place_forget()
    window.title("Verify")

    frame5 = Frame(window, height = 100, width=100, bg= "skyblue", bd=1)
    frame5.place(anchor = CENTER, relx = 0.5, rely = 0.50)

    takebutton=Button(frame5, text="Take\n Palm Photo",bg="dodgerblue4",fg='white',
                       activebackground='steelblue',activeforeground='white',font=myFont,
                       command = lambda : take_photo(takebutton, backbutton,frame5), height=2, width=10)
    takebutton.pack(pady=(100,0), padx=(100,100), anchor=CENTER)

    backbutton=Button(frame5, text="Back", font=myFont,bg="dodgerblue4", fg='white',
                       activebackground='steelblue',activeforeground='white',
                       command = lambda : homepage2(takebutton, backbutton,frame5), height=2, width=10)
    backbutton.pack(pady=(20,100), padx=(100,100), anchor=CENTER)

def take_photo(takebutton, backbutton,frame5):
    takebutton.pack_forget()
    backbutton.pack_forget()
    frame5.place_forget()
    window.title("Capturing Right")   # NOTE: copy-pasted from right_cap, never renamed
    cam = cv2.VideoCapture(0)
    print("take photo")
    global start
    program_starts = time.time()
    while(True):
        now = time.time()
        targetTemp = "{:.1f}".format(mlx.object_temperature)
        target_temp = checktest(targetTemp)

        _, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Verify Image", gray)
        target_temp = checktest(targetTemp)
        print("Target Temperature:", target_temp, "\u00b0C")

        k = cv2.waitKey(1)
        if now-program_starts >= 4:
            break

    for i in range(1):
        _, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Verify Image", gray)
        k = cv2.waitKey(1)
        cv2.imwrite(temp_folder + "test.jpg", gray)

    cam.release()
    cv2.destroyAllWindows()
    cap_img = temp_folder + "test.jpg"
    waiting_process(cap_img,target_temp)

def take_photo2(takebutton, backbutton,frame5):
    # NOTE: dead code - never called anywhere, and references an undefined
    # global `cam`. Left in exactly as found in the original.
    takebutton.pack_forget()
    backbutton.pack_forget()
    frame5.place_forget()
    print("Take Photo")
    program_starts = time.time()
    while(True):
        now = time.time()
        _, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Verify Image", gray)
        k = cv2.waitKey(1)
        if now-program_starts >= 4:
            break

    for i in range(1):
        _, frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Verify Image", gray)
        k = cv2.waitKey(1)
        cv2.imwrite(temp_folder + "test.jpg", gray)

    cam.release()
    cv2.destroyAllWindows()
    cap_img = temp_folder + "test.jpg"
    waiting_process(cap_img)

def waiting_process(cap_img,target_temp):
    print("image pre process 2")
    waitlabel = Label(window, text="Waiting")
    waitlabel.pack(pady=(10), anchor=CENTER)

    img_original = cv2.imread(cap_img, cv2.IMREAD_COLOR)
    #ROI
    #x,y,w,h = 270,117,159,86
    x,y,w,h = 273,94,214,195
    roi = img_original[y:y+h,x:x+w]
    print("ROI")
    # + noise
    noiseReduced = cv2.fastNlMeansDenoising(roi)
    print("Noise")

    #CLAHE
    kernel = np.ones((7,7),np.uint8)
    img = cv2.morphologyEx(noiseReduced, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    frame = cv2.bitwise_not(img_output)
    print("CLAHE")

    #Skeletonize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    ret,frame2 = cv2.threshold(skel, 5, 255, cv2.THRESH_BINARY)
    print("Skeleton")
    x1,y1,w1,h1 = 4,4,205,187
    newframe = frame2[y1:y1+h1,x1:x1+w1]

    waiting_process_testing(newframe,waitlabel,target_temp)

def checktest(targetTemp):
    new_temp = float("%.2f"%float(targetTemp)) + 1.0
    return new_temp

def waiting_process_testing(newframe,waitlabel,target_temp):
    print("image testing2")
    waitlabel.pack_forget()

    new_model = keras.models.load_model(h5_model)

    pyplot.imsave(temp_folder + "test.jpg", newframe, cmap='gray')
    img = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB)   # NOTE: newframe is single-channel by this point
    img_array = Image.fromarray(img, 'RGB')
    resized_img = img_array.resize((224, 224))
    images = np.array(resized_img)
    test_image = images.reshape(1,224,224,3)
    test_image2 = test_image.astype('float32')/255.

    predicted_cat = new_model.predict(test_image2).argmax()
    new_cat = new_model.predict(test_image2)   # NOTE: predicts twice - original did this too
    list_cat = new_cat.flatten().tolist()
    percent = list_cat[predicted_cat]
    print("Percent: " + str(percent))
    print("Class: " + str(predicted_cat))
    print("List Proba: " + str(list_cat))

    if percent < 0.80:
        window.after(1000, lambda:alt_output())
    else:
        counter = 0
        for i in range(predicted_cat):
            if i%2 == 1:
                counter+=1
        category = sorted(os.listdir(main_folder))
        a = category
        id_num = a[counter]
        window.after(1000, lambda:checkk(id_num,target_temp))

def alt_output():
    frame8 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame8.place(anchor = CENTER, relx = 0.5, rely = 0.50)
    verlabel = Label(frame8,bg='skyblue', fg='black', font=("bold", 24),
                      text="No existing matches found\nPlease try again later")
    verlabel.pack(pady=(10), anchor=CENTER)
    window.after(5000, lambda:homepage4(frame8, verlabel))

#verify data
def checkk(id_num,target_temp):
    eid = 10
    idd= id_num
    temperature = target_temp

    mysqldb=mysql.connector.connect(host=host_id,user="root",password="",database="tracking_db")
    mycursor=mysqldb.cursor()

    try:
        temporary="INSERT INTO temporary (establishment_id,temperature,tracking_id) VALUES (%s , %s, %s)"
        val = (eid,temperature,idd)
        mycursor.execute(temporary,val)
        mysqldb.commit()
        lastid = mycursor.lastrowid

        mycursor.execute("\
            INSERT INTO person_tracks(person_id, establishment_id,temperature, tracking_id)\
            SELECT persons.id, temporary.establishment_id,temporary.temperature, temporary.tracking_id\
            FROM persons\
            JOIN temporary\
            ON persons.tracking_id = temporary.tracking_id\
            WHERE temporary.tracking_id = '" + str(idd)+ "'")
        mysqldb.commit()
        lastid = mycursor.lastrowid

        mycursor.execute("DELETE FROM temporary WHERE temporary.tracking_id = '" + str(idd)+ "'")
        mysqldb.commit()
        lastid = mycursor.lastrowid
        print("in")
    except Exception as e:
        print(e)
        mysqldb.rollback()
        mysqldb.close()
    # NOTE: mycursor/mysqldb are reused below even on the exception path,
    # where mysqldb.close() was just called above - that's how it was.

    sql = mycursor.execute("SELECT persons.firstname, persons.middlename, persons.lastname, \
        person_tracks.temperature, person_tracks.date_created\
        FROM persons\
        JOIN person_tracks\
        ON persons.tracking_id = person_tracks.tracking_id\
        WHERE persons.tracking_id = '" + str(idd) + "' \
        ORDER BY person_tracks.date_created DESC LIMIT 1")

    result = mycursor.fetchone()
    lastid = mycursor.lastrowid

    print("Name: " + result[0] + " " + result[1] + " " + result[2])
    print("Temperature: " + result[3])
    print("Date/Time: " + str(result[4]))

    frame8 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame8.place(anchor = CENTER, relx = 0.5, rely = 0.50)
    verlabel = Label(frame8,bg='skyblue', fg='black', font=("bold", 24), text="Successfully Verified\nWelcome")
    verlabel.pack(pady=(10), anchor=CENTER)

    fullnamelabel = Label(frame8,bg='skyblue', fg='black', font=("bold", 16),
                           text="Name: " + result[0] + " " + result[1] + " " + result[2])
    fullnamelabel.pack(pady=(10), anchor=CENTER)

    templabel = Label(frame8,bg='skyblue', fg='black', font=("bold", 16), text="Temperature: " + result[3])
    templabel.pack(pady=(10), anchor=CENTER)

    datelabel = Label(frame8,bg='skyblue', fg='black', font=("bold", 16), text="Date/Time: " + str(result[4]))
    datelabel.pack(pady=(10), anchor=CENTER)

    global start
    global end
    end = timeit.default_timer()
    print('Time: ', end - start)
    start,end =0,0   # NOTE: reset right after printing, so this "elapsed time" was never really meaningful
    window.after(5000, lambda:homepage3(verlabel, fullnamelabel, templabel, datelabel))

def BackAdd(backadd,entry_last,entry_first,entry_middle,sex,entry_contact,entry_email,entry_address,entry_province,entry_zipcode,
            human, frame4, label_0,lastname,firstname,middlename,droplist,contact,email,address,province,zipcode,addbutton):
    # NOTE: see how this is *called* below in register() - the argument order
    # there doesn't match this parameter order at all. It "works" purely by
    # accident, because every branch just calls .place_forget() on whatever
    # it's handed, regardless of what it's named here.
    human.place_forget()
    backadd.place_forget()
    entry_last.place_forget()
    entry_first.place_forget()
    entry_middle.place_forget()
    sex.place_forget()
    entry_contact.place_forget()
    entry_email.place_forget()
    entry_zipcode.place_forget()
    label_0.place_forget()
    lastname.place_forget()
    firstname.place_forget()
    middlename.place_forget()
    droplist.place_forget()
    contact.place_forget()
    email.place_forget()
    address.place_forget()
    province.place_forget()
    zipcode.place_forget()
    entry_address.place_forget()
    entry_province.place_forget()
    addbutton.place_forget()
    frame4.place_forget()

    frame1 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
    frame1.place(anchor = CENTER, relx = 0.5, rely = 0.50)

    title1 = Label(window, text="DEVELOPMENT OF PALM VEIN BIOMETRICS \nWITH TEMPERATURE SCANNER \n FOR CONTACT TRACING APPLICATION",
                   width=50, height=4, bg='white', fg='black',anchor=CENTER,font=("bold", 32))
    title1.place(anchor = CENTER, relx = 0.5, rely = 0.12)

    #register
    regbutton=Button(frame1, text="REGISTER\n USER",font=myFont, bg="dodgerblue4", fg='white',
                      activebackground='steelblue',activeforeground='white',width=10,borderwidth = 0,
                      command = lambda : register(regbutton, verifybutton, exitbutton,frame1,title1))
    regbutton.pack(anchor = CENTER, side = LEFT, pady=100,padx=(150,50))

    #verify
    verifybutton=Button(frame1,text="VERIFY\n USER", font=myFont,width=10,bg="dodgerblue4", fg='white',
                         activebackground='steelblue',activeforeground='white', borderwidth = 0,
                         command = lambda : verify_real(regbutton, verifybutton, exitbutton, frame1, title1))
    verifybutton.pack(anchor = CENTER, side = LEFT,pady=100,padx=(25,19))

    #exit
    exitbutton=Button(frame1, text="EXIT\n PROGRAM",font=myFont,width=10,bg="dodgerblue4", fg='white',
                       activebackground='steelblue',activeforeground='white', borderwidth = 0, command = exit)
    exitbutton.pack(anchor = CENTER, side = RIGHT,pady=100,padx=(50,150))

#####################Register#####################
def register(regbutton, verifybutton, exitbutton,frame1,title1):
    regbutton.pack_forget()
    verifybutton.pack_forget()
    exitbutton.pack_forget()
    frame1.place_forget()
    title1.place_forget()
    window.title("Register")

    label_0 = Label(window, text="USER REGISTRATION FORM",width=32, height=1, bg='dodgerblue2', fg='white',
                     anchor=CENTER,font=("bold", 32))
    label_0.place(anchor = CENTER, relx = 0.5, rely = 0.05)

    frame4=Frame(window, height = 600, width=800, bg= "skyblue", bd=1, relief = FLAT)
    frame4.place(anchor = CENTER, relx = 0.5, rely = 0.52)

    #id
    krazynum = StringVar(value = str(uuid.uuid4().int)[0:5])

    lastname = Label(frame4, text="Last Name",width=10,font=("bold", 15))
    lastname.place(anchor = CENTER, relx = 0.20, rely = 0.12)
    entry_last = Entry(frame4,width=18,font=("bold", 20))
    entry_last.place(anchor = CENTER, relx = 0.20, rely = 0.05)

    firstname = Label(frame4, text="First Name",width=10,font=("bold", 15))
    firstname.place(anchor = CENTER, relx = 0.60, rely = 0.12)
    entry_first = Entry(frame4,width=18,font=("bold", 20))
    entry_first.place(anchor = CENTER, relx = 0.60, rely = 0.05)

    middlename = Label(frame4, text="M.I.",width=4,font=("bold", 15))
    middlename.place(anchor = CENTER, relx = 0.88, rely = 0.12)
    entry_middle = Entry(frame4,width=4,font=("bold", 20))
    entry_middle.place(anchor = CENTER, relx = 0.88, rely = 0.05)

    zipcode = Label(frame4, text="Age",width=4,font=("bold", 15))   # NOTE: labeled "Age" but variable named zipcode
    zipcode.place(anchor = CENTER, relx = 0.15, rely = 0.27)
    entry_zipcode = Entry(frame4,width=4,font=("bold", 20))
    entry_zipcode.place(anchor = CENTER, relx = 0.15, rely = 0.20)

    sex = Label(frame4, text="Gender",width=10,font=("bold", 15))
    sex.place(anchor = CENTER, relx = 0.37, rely = 0.27)
    var = StringVar()
    list1 = ['MALE','FEMALE']
    droplist=OptionMenu(window,var, *list1)   # NOTE: master is `window`, not `frame4`, unlike everything else here
    droplist.config(width=14,font=("bold", 16))
    droplist.place(anchor = CENTER, relx = 0.42, rely = 0.27)
    var.set('MALE')

    contact = Label(frame4, text="Contact",width=10,font=("bold", 15))
    contact.place(anchor = CENTER, relx = 0.75, rely = 0.27)
    entry_contact = Entry(frame4,width=15,font=("bold", 20))
    entry_contact.place(anchor = CENTER, relx = 0.75, rely = 0.20)

    email = Label(frame4, text="Email",width=10,font=("bold", 15))
    email.place(anchor = CENTER, relx = 0.50, rely = 0.45)
    entry_email = Entry(frame4,width=35,font=("bold", 20))
    entry_email.place(anchor = CENTER, relx = 0.50, rely = 0.37)

    address = Label(frame4, text="Address",width=10,font=("bold", 15))
    address.place(anchor = CENTER, relx = 0.50, rely = 0.62)
    entry_address = Entry(frame4,width=30,font=("bold", 20))
    entry_address.place(anchor = CENTER, relx = 0.50, rely = 0.55)

    province = Label(frame4, text="Province",width=10,font=("bold", 15))
    province.place(anchor = CENTER, relx =0.50, rely = 0.79)
    entry_province = Entry(frame4,width=10,font=("bold", 20))
    entry_province.place(anchor = CENTER, relx = 0.50, rely = 0.72)

    backadd = Button(frame4, text='BACK', font=("bold", 20),activebackground='steelblue',activeforeground='white',
                      command = lambda :BackAdd(frame4, backadd,entry_last,entry_first,entry_middle,sex,entry_contact,entry_email,
                                                 entry_address,entry_province,entry_zipcode,
                                                 label_0,lastname,backadd,firstname,middlename,droplist,contact,email,address,province,
                                                 zipcode,addbutton),
                      width=10,bg='dodgerblue4',fg='white')
    backadd.place(anchor = CENTER, relx = 0.35, rely = 0.90)

    addbutton=Button(frame4, text='NEXT', font=("bold", 20),activebackground='steelblue',activeforeground='white',
                      command = lambda :Add(entry_last,entry_first,entry_middle,var,sex,entry_contact,entry_email,entry_address,
                                             entry_province,entry_zipcode,krazynum,
                                             label_0,lastname,firstname,middlename,droplist,contact,email,address,province,zipcode,
                                             addbutton,backadd,frame4),
                      width=10,bg='dodgerblue4',fg='white')
    addbutton.place(anchor = CENTER, relx = 0.65, rely = 0.90)

################### Initial Window ####################
window = tk.Tk()
window.configure(background='white')

myFont=font.Font(family="Helvetica", size=36, weight="bold")
myFont2=font.Font(family="Helvetica", size=18, weight="bold")
window.title("Home")

i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)

frame1 = Frame(window, height = 300, width=450, bg= "skyblue", bd=1)
frame1.place(anchor = CENTER, relx = 0.5, rely = 0.50)

title1 = Label(window, text="DEVELOPMENT OF PALM VEIN BIOMETRICS \nWITH TEMPERATURE SCANNER \n FOR CONTACT TRACING APPLICATION",
               width=50, height=4, bg='white', fg='black',anchor=CENTER,font=("bold", 32))
title1.place(anchor = CENTER, relx = 0.5, rely = 0.12)

pad=3
window.geometry("1250x720")
window.resizable(0, 0)

#register
regbutton=Button(frame1, text="REGISTER\n USER",font=myFont, bg="dodgerblue4", fg='white',
                  activebackground='steelblue',activeforeground='white', width=10,
                  command = lambda : register(regbutton, verifybutton, exitbutton,frame1,title1))
regbutton.pack(anchor = CENTER, side = LEFT, pady=100,padx=(150,50))

#verify
verifybutton=Button(frame1,text="VERIFY\n USER", font=myFont,width=10,bg="dodgerblue4", fg='white',
                     activebackground='steelblue',activeforeground='white',
                     command = lambda : verify_real(regbutton, verifybutton, exitbutton, frame1, title1))
verifybutton.pack(anchor = CENTER, side = LEFT,pady=100,padx=(25,19))

#exit
exitbutton=Button(frame1, text="EXIT\n PROGRAM",font=myFont,width=10,bg="dodgerblue4", fg='white',
                   activebackground='steelblue',activeforeground='white',command = exit)
exitbutton.pack(anchor = CENTER, side = RIGHT,pady=100,padx=(50,150))

window.mainloop()
