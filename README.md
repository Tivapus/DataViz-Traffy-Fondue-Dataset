# Data Visualization: Traffy Fondue Dataset 📊

โปรเจกต์นี้เป็นส่วนหนึ่งของรายวิชา **2110446 Data Science and Data Engineering** โดยมีวัตถุประสงค์เพื่อนำชุดข้อมูล "Traffy Fondue" มาวิเคราะห์และนำเสนอในรูปแบบของ Data Visualization

---

## 🏙️ เกี่ยวกับชุดข้อมูล (About The Dataset)

**Traffy Fondue (ทราฟฟี่ฟองดูว์)** คือแพลตฟอร์มที่เปิดให้ประชาชนสามารถแจ้งปัญหาของเมืองที่พบเจอ เช่น ปัญหาความสะอาด, ทางเท้าชำรุด, ไฟฟ้าสาธารณะ, และอื่นๆ เพื่อให้หน่วยงานที่รับผิดชอบเข้ามาดำเนินการแก้ไขได้อย่างรวดเร็ว

ชุดข้อมูลที่ใช้ในโปรเจกต์นี้รวบรวมมาจากปัญหาต่างๆ ที่ประชาชนได้แจ้งเรื่องเข้ามา ซึ่งเป็นข้อมูลจริงที่สะท้อนถึงปัญหาในพื้นที่ต่างๆ ได้เป็นอย่างดี

---

## 🎨 ส่วนที่รับผิดชอบ (My Contribution)

ในโปรเจกต์นี้ ผมได้รับหน้าที่หลักในส่วนของ **Data Visualization** ซึ่งครอบคลุมตั้งแต่:

- การสำรวจและวิเคราะห์ข้อมูลเบื้องต้น (Exploratory Data Analysis)
- การทำความสะอาดข้อมูล (Data Cleaning) และการสร้าง Feature เพิ่มเติม (Feature Engineering)
- การออกแบบและสร้าง Dashboard แบบ Interactive เพื่อนำเสนอผลการวิเคราะห์ในรูปแบบที่เข้าใจง่ายและสวยงาม

---

## 📁 ไฟล์ในโปรเจกต์ (Project Files)

1.  `EDA วิเคราะห์ดาต้าก่อนทำทุกอย่างเลย (pre process).ipynb`

    - Notebook สำหรับการสำรวจและวิเคราะห์ข้อมูลดิบ (Exploratory Data Analysis) ในขั้นตอนแรก เพื่อทำความเข้าใจภาพรวมของข้อมูลทั้งหมดก่อนนำไปประมวลผลต่อ

2.  `EDA หลัง clean data + Add column เพิ่มเติม แล้ว.ipynb`

    - Notebook ที่แสดงผลการวิเคราะห์ข้อมูลหลังจากผ่านกระบวนการทำความสะอาด (Data Cleaning) และมีการสร้างคอลัมน์ใหม่ๆ ที่เป็นประโยชน์ต่อการวิเคราะห์ในเชิงลึก

3.  `Data Visualization Dashboard.py`
    - สคริปต์ภาษา Python ที่ใช้ไลบรารี **Streamlit** ในการสร้าง Dashboard แบบ Interactive สำหรับนำเสนอข้อมูลเชิงภาพ

---

## 🛠️ เทคโนโลยีที่ใช้ (Technologies Used)

- **Python**
- **Pandas:** สำหรับจัดการและวิเคราะห์ข้อมูล
- **Matplotlib & Seaborn:** สำหรับสร้างกราฟและแผนภูมิต่างๆ
- **Streamlit:** สำหรับสร้าง Web Application และ Dashboard

---

## 🚀 วิธีการรัน Dashboard (How to Run)

1.  Clone a copy of the repository
    ```bash
    git clone [https://github.com/Tivapus/DataViz-Traffy-Fondue-Dataset.git](https://github.com/Tivapus/DataViz-Traffy-Fondue-Dataset.git)
    ```
2.  ติดตั้ง Dependencies ที่จำเป็น (แนะนำให้สร้าง `requirements.txt`)
    ```bash
    pip install pandas matplotlib seaborn streamlit
    ```
3.  รัน Dashboard ผ่าน Streamlit
    ```bash
    streamlit run "Data Visualization Dashboard.py"
    ```

---

## 👤 ผู้จัดทำ (Author)

- **Tivapus**
