import streamlit as st
import torch
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
import math
from ultralytics import YOLO
import time
model=YOLO('classy.pt')
st.set_page_config(page_title="TumorAi App")
# dengue fever

gcauses = """
The exact causes of glioma, a type of brain tumor, are not fully understood. However, certain risk factors have been identified. These include exposure to ionizing radiation, a family history of glioma, and certain genetic disorders such as neurofibromatosis type 1 and Li-Fraumeni syndrome. While these factors may increase the risk, in many cases, the underlying cause of glioma remains unknown.
"""
geffects = """
Gliomas can have significant effects on brain function and overall health. As the tumor grows, it can exert pressure on surrounding brain tissue, leading to symptoms such as headaches, seizures, difficulty speaking or understanding language, memory problems, changes in personality or mood, and neurological deficits like weakness or loss of sensation in the limbs. The severity and specific symptoms experienced by an individual can vary depending on the location, size, and grade of the glioma.
"""
gtreat = """
The treatment of glioma depends on several factors, including the tumor's location, size, grade, and the patient's overall health. Treatment options may include surgery to remove the tumor, radiation therapy to target and kill cancer cells, and chemotherapy to destroy or slow down tumor growth. In some cases, a combination of these treatments may be used. The choice of treatment is determined by a multidisciplinary team of medical professionals and is tailored to the individual patient's needs and circumstances. Regular monitoring and follow-up care are essential to assess the tumor's response to treatment and manage any potential side effects.
"""

mcauses = """The exact causes of meningioma, a type of brain tumor, are not well understood. However, certain risk factors have been identified, including radiation exposure, such as previous radiation therapy to the head, and certain genetic conditions like neurofibromatosis type 2 (NF2). Hormonal factors, such as increased levels of estrogen, have also been associated with an increased risk of developing meningiomas. Nonetheless, the underlying cause of most meningiomas remains unknown.
"""
meffects = """Meningiomas can have varying effects depending on their size, location, and growth rate. Some meningiomas may not cause noticeable symptoms and can be incidentally discovered during imaging tests conducted for unrelated reasons. However, when symptoms do occur, they can include headaches, seizures, changes in vision or hearing, weakness or numbness in the limbs, and cognitive or personality changes. The specific symptoms and their severity can differ from person to person.
"""
mtreat = """
The treatment of meningioma depends on factors such as tumor size, location, and growth rate, as well as the individual's overall health. Treatment options may include observation with regular monitoring for slow-growing or asymptomatic tumors, surgery to remove the tumor, radiation therapy to target and destroy cancer cells, and in some cases, medication to manage symptoms or slow down tumor growth. The choice of treatment is based on a thorough evaluation by a multidisciplinary team of healthcare professionals and is tailored to the specific needs of each patient. Regular follow-up care is important to assess the tumor's response to treatment and address any potential complications or recurrence.
"""

pcauses = """
The exact causes of pituitary tumors, also known as pituitary adenomas, are not fully understood. However, certain factors may increase the risk of their development. These include genetic conditions like multiple endocrine neoplasia type 1 (MEN1) and Carney complex, as well as rare hereditary syndromes such as familial isolated pituitary adenoma. Hormonal imbalances, exposure to certain chemicals, and head injuries have also been suggested as potential contributing factors. However, in many cases, the underlying cause of pituitary tumors remains unknown."""
peffects = """Pituitary tumors can have diverse effects depending on their size, location, and hormone production. They can disrupt the normal functioning of the pituitary gland, leading to hormonal imbalances and associated symptoms. The specific effects can vary widely, ranging from vision problems and headaches due to pressure on nearby structures, to hormonal disturbances resulting in issues such as infertility, growth abnormalities, changes in body composition, and metabolic problems. The effects of pituitary tumors are highly dependent on the specific hormones involved and the individual's overall health.
"""
ptreat = """The treatment of pituitary tumors depends on several factors, including the tumor's size, hormone production, and the individual's overall health. Treatment options may include medication to regulate hormone levels, surgery to remove the tumor, radiation therapy to destroy tumor cells, or a combination of these approaches. The choice of treatment is determined by a multidisciplinary team of medical professionals and is tailored to the individual patient's needs and circumstances. Regular monitoring and follow-up care are often necessary to manage hormone levels, monitor tumor growth, and ensure optimal treatment outcomes.
"""
st.markdown(
    """
        <style>
            [data-testid="stSidebarNav"] {
                background-repeat: no-repeat;                
            }
            [data-testid="stSidebarNav"]::before {
                content: "TumorAi";
                margin-left: 20px;
                margin-top: 20px;

                font-size: 30px;
                text-align: center;
                position: relative;
            }
        </style>
        """,
    unsafe_allow_html=True,
)
st.title("TumorAi Application")
st.text(
    "Upload an image of a close up of a tumerous MRI scan and we will tell you what type it is."
)
# read images.zip as a binary file and put it into the button
with open("test.zip", "rb") as fp:
    btn = st.download_button(
        label="Download test images",
        data=fp,
        file_name="test.zip",
        mime="application/zip",
    )
image = st.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False
)

if image:
    disp = False
    image = Image.open(image)
    model = YOLO("classy.pt")
    
    results = model(image)
    disp = True

    c = st.image("loader.gif")
    time.sleep(3)
    c.empty()
    print(type(results[0].probs))
    print(results[0].probs)
    resy = results[0].probs.tolist()
    fresy = [ '%.2f' % elem for elem in resy ]
    # st.text(fresy)
    
    fig = px.imshow(np.squeeze(image), aspect="equal")
    st.plotly_chart(fig)
    
    # st.text(results.pandas().xyxy)
    na = fresy.index(max(fresy))
    if na == 0:
        name = "Glioma"
    elif na == 1:
        name = "Meningioma"
    elif na == 2:
        name = "Pituitary"

    if name:
        st.text(f"Detected {name} with high confidence")
        if name == "Glioma":
            st.write(
                """
                Glioma is a brain tumor that develops from glial cells. Its exact causes are not fully known, but risk factors include radiation exposure and certain genetic disorders. Gliomas can affect brain function, causing headaches, seizures, and neurological deficits. MRI is used to detect and evaluate gliomas, showing abnormal masses or areas of increased signal intensity. The size, location, and enhancement pattern of the tumor help determine its grade and guide treatment decisions.
                """
            )
            st.image("images/glioma.webp")
            st.write("More Info")

            tab1, tab2, tab3 = st.tabs(
                ["Causes", "Effects", "Treatment"]
            )
            with tab1:
                st.write(gcauses)
                st.write(
                    "More Info can be found on the [Mayo clinic website](https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251)"
                )
            with tab2:
                st.write(geffects)
                st.write(
                    "More Info can be found on the [Mayo clinic website](https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251)"
                )
            with tab3:
                st.write(gtreat)
                st.write(
                    "More Info can be found on the [Mayo clinic website](https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251)"
                )
        elif (
            name == "Meningioma"
        ):
            st.write(
                """
                Meningioma is a brain tumor that originates from the meninges, the protective membranes covering the brain and spinal cord. Its exact cause is unknown, but risk factors include radiation exposure, certain genetic conditions, and hormonal factors. Meningiomas can vary in symptoms depending on size and location. MRI is commonly used to detect and evaluate meningiomas, showing well-defined masses with a dural tail.
                """
            )
            st.image("images/Meningioma.jfif")
            st.write("Known Carried Diseases")
            btab1, btab2, btab3 = st.tabs(
                ["Causes", "Effects", "Treatment"]
            )
            with btab1:
                st.write(mcauses)
                st.write(
                    "More Info can be found on the [Cancer Website](https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma)"
                )
            with btab2:
                st.write(meffects)
                st.write(
                    "More Info can be found on the [Cancer Website](https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma)"
                )
            with btab3:
                st.write(mtreat)
                st.write(
                    "More Info can be found on the [Cancer Website](https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma)"
                )
        elif name == "Pituitary":
            st.write(
                """
                A pituitary tumor, also known as pituitary adenoma, is a non-cancerous growth in the pituitary gland. It can be functioning or non-functioning, causing hormonal imbalances or symptoms due to its size. Symptoms may include headaches, vision problems, fatigue, and hormonal disturbances. Diagnosis involves imaging tests like MRI, and treatment options include medication, surgery, or radiation therapy.
                """
            )
            st.image("images/petu.jfif")
            st.write("Known Carried Diseases")
            ctab1, ctab2, ctab3 = st.tabs(
                ["Causes", "Effects", "Treatment"]
            )
            with ctab1:
                st.write(pcauses)
                st.write(
                    "More Info can be found on the [MAYO clinic website](https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350548)"
                )
            with ctab2:
                st.write(peffects)
                st.write(
                    "More Info can be found on the [MAYO clinic website](https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350548)"
                )
            with ctab3:
                st.write(ptreat)
                st.write(
                    "More Info can be found on the [MAYO clinic website](https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350548)"
                )
