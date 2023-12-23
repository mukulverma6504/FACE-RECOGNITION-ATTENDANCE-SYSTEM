import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
 
jobs_image = face_recognition.load_image_file("D:\Project-1-main\python project/jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
 
ratan_tata_image = face_recognition.load_image_file("D:\Project-1-main\python project/tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]
 
modi_image = face_recognition.load_image_file("D:\Project-1-main\python project/modi.jpg")
modi_encoding = face_recognition.face_encodings(modi_image)[0]
 
tesla_image = face_recognition.load_image_file("D:\Project-1-main\python project/tesla.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

mukul_image = face_recognition.load_image_file("D:\Project-1-main\python project/mukul.jpg")
mukul_encoding = face_recognition.face_encodings(mukul_image)[0]

ayaan_image = face_recognition.load_image_file("D:\Project-1-main\python project/ayaan.jpg")
ayaan_encoding = face_recognition.face_encodings(ayaan_image)[0]

harsh_image = face_recognition.load_image_file("D:\Project-1-main\python project/harsh.jpg")
harsh_encoding = face_recognition.face_encodings(harsh_image)[0]

dhruv_image = face_recognition.load_image_file("D:\Project-1-main\python project/dhruv.jpg")
dhruv_encoding = face_recognition.face_encodings(dhruv_image)[0]

mrityunjay_image = face_recognition.load_image_file("D:\Project-1-main\python project/mrityunjay.jpg")
mrityunjay_encoding = face_recognition.face_encodings(mrityunjay_image)[0]

manish_image = face_recognition.load_image_file("D:\Project-1-main\python project/manish.jpg")
manish_encoding = face_recognition.face_encodings(manish_image)[0]

Monika_mam_image = face_recognition.load_image_file("D:\Project-1-main\python project/Monika_mam.jpg")
Monika_mam_encoding = face_recognition.face_encodings(Monika_mam_image)[0]

kunal_image = face_recognition.load_image_file("D:\Project-1-main\python project/kunal.jpg")
kunal_encoding = face_recognition.face_encodings(kunal_image)[0]

Adarsh_image = face_recognition.load_image_file("D:\Project-1-main\python project/Adarsh.jpg")
Adarsh_encoding = face_recognition.face_encodings(Adarsh_image)[0]

Anshu_image = face_recognition.load_image_file("D:\Project-1-main\python project/Anshu.jpg")
Anshu_encoding = face_recognition.face_encodings(Anshu_image)[0]

Ashish_image = face_recognition.load_image_file("D:\Project-1-main\python project/Ashish.png")
Ashish_encoding = face_recognition.face_encodings(Ashish_image)[0]

Imbesat_image = face_recognition.load_image_file("D:\Project-1-main\python project/Imbesat.jpg")
Imbesat_encoding = face_recognition.face_encodings(Imbesat_image)[0]

ishmeat_image = face_recognition.load_image_file("D:\Project-1-main\python project/ishmeat.jpg")
ishmeat_encoding = face_recognition.face_encodings(ishmeat_image)[0]

manisha_image = face_recognition.load_image_file("D:\Project-1-main\python project/manisha.jpg")
manisha_encoding = face_recognition.face_encodings(manisha_image)[0]

Priyansh_image = face_recognition.load_image_file("D:\Project-1-main\python project/Priyansh.jpg")
Priyansh_encoding = face_recognition.face_encodings(Priyansh_image)[0]

puneet_image = face_recognition.load_image_file("D:\Project-1-main\python project/puneet.jpg")
puneet_encoding = face_recognition.face_encodings(puneet_image)[0]

Riya_image = face_recognition.load_image_file("D:\Project-1-main\python project/Riya.jpg")
Riya_encoding = face_recognition.face_encodings(Riya_image)[0]

shivani_image = face_recognition.load_image_file("D:\Project-1-main\python project/shivani.jpg")
shivani_encoding = face_recognition.face_encodings(shivani_image)[0]

Anand_image = face_recognition.load_image_file("D:\Project-1-main\python project/Anand.jpg")
Anand_encoding = face_recognition.face_encodings(Anand_image)[0]

shubham_image = face_recognition.load_image_file("D:\Project-1-main\python project/shubham.jpg")
shubham_encoding = face_recognition.face_encodings(shubham_image)[0]

sugandh_image = face_recognition.load_image_file("D:\Project-1-main\python project/sugandh.jpg")
sugandh_encoding = face_recognition.face_encodings(sugandh_image)[0]

vansh_image = face_recognition.load_image_file("D:\Project-1-main\python project/vansh.png")
vansh_encoding = face_recognition.face_encodings(vansh_image)[0]

kanishka_image = face_recognition.load_image_file("D:\Project-1-main\python project/kanishka.jpg")
kanishka_encoding = face_recognition.face_encodings(kanishka_image)[0]

payal_mam_image = face_recognition.load_image_file("D:\Project-1-main\python project/payal_mam.jpg")
payal_mam_encoding = face_recognition.face_encodings(payal_mam_image)[0]

megha_mam_image = face_recognition.load_image_file("D:\Project-1-main\python project/megha_mam.jpg")
megha_mam_encoding = face_recognition.face_encodings(megha_mam_image)[0]

harshita_mam_image = face_recognition.load_image_file("D:\Project-1-main\python project/harshita_mam.jpg")
harshita_mam_encoding = face_recognition.face_encodings(harshita_mam_image)[0]

sangita_mam_image = face_recognition.load_image_file("D:\Project-1-main\python project/sangita_mam.jpg")
sangita_mam_encoding = face_recognition.face_encodings(sangita_mam_image)[0] 

ashish_sir_image = face_recognition.load_image_file("D:\Project-1-main\python project/ashish_sir.jpg")
ashish_sir_encoding = face_recognition.face_encodings(ashish_sir_image)[0] 



 
known_face_encoding = [
jobs_encoding,
ratan_tata_encoding,
modi_encoding,
mukul_encoding,
tesla_encoding,
ayaan_encoding,
harsh_encoding,
dhruv_encoding,
mrityunjay_encoding,
manish_encoding,
Monika_mam_encoding,
kunal_encoding,
Adarsh_encoding,
Anshu_encoding,
Ashish_encoding,
Imbesat_encoding,
ishmeat_encoding,
manisha_encoding,
Priyansh_encoding,
puneet_encoding,
Riya_encoding,
shivani_encoding,
Anand_encoding,
shubham_encoding,
sugandh_encoding,
vansh_encoding,
kanishka_encoding,
payal_mam_encoding,
megha_mam_encoding,
harshita_mam_encoding,
sangita_mam_encoding,
ashish_sir_encoding
]
 
known_faces_names = [
"Jobs",
"Ratan tata",
"Modi",
"Mukul",
"Tesla",
"Ayaan",
"Harsh",
"Dhruv",
"Mrityunjay",
"manish",
"Monika_mam",
"kunal",
"Adarsh",
"Anshu",
"Ashish",
"Imbesat",
"ishmeat",
"manisha",
"Priyansh",
"puneet",
"Riya",
"shivani",
"Anand",
"shubham",
"sugandh",
"vansh",
"kanishka",
"payal_mam",
"megha_mam",
"harshita_mam",
"sangita_mam",
"ashish_sir"
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font,   +++











                        +++++++++++++++++++
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%I:%M:%S %p")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()
