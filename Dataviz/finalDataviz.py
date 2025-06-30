import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# SETUP PAGE
# -------------------------------
st.set_page_config(page_title="Bangkok Analysis", layout="wide")
st.title('Traffy Fondue Dataset')
tab1, tab2, tab3 = st.tabs(["\U0001F4CA Overview", "\U0001F3ED District Performance", "\U0001F916 Model Results"])

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('data/processed_traffy.csv')
    data['coords'] = data['coords'].str.strip('()')
    data[['longitude', 'latitude']] = data['coords'].str.split(',', expand=True).astype(float)
    data.drop(columns='coords', inplace=True)
    data = data[data['state'] == 'เสร็จสิ้น']
    data = data[data['type'].isin(['{ท่อระบายน้ำ}', '{ทางเท้า}', '{แสงสว่าง}', '{กีดขวาง}', '{ต้นไม้}'])]
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', utc=True)
    data['last_activity'] = pd.to_datetime(data['last_activity'], format='mixed', utc=True)

    with open("data/Bangkok-districts.geojson", encoding="utf-8") as f:
        geojson_data = json.load(f)

    results = pd.read_csv('data/result_total.csv')
    return data, geojson_data, results

df, geojson_data, results = load_data()

# -------------------------------
# SETUP GEOJSON COLOR LAYER
# -------------------------------
for feature in geojson_data["features"]:
    feature["properties"]["fill_color"] = [
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(100, 255),
        30
    ]

geojson_layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson_data,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[0, 0, 0, 80],
    auto_highlight=True,
)

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

show_total = st.sidebar.checkbox("รวมทั้งหมด", value=False)
selected_types = st.sidebar.multiselect("เลือกประเภท (type)", df['type'].unique())
selected_districts = st.sidebar.multiselect("เลือกเขต (district)", df['district'].unique())
start_date, end_date = st.sidebar.date_input("เลือกช่วงวันที่", [min_date, max_date])

# -------------------------------
# PROCESSING
# -------------------------------
df['time2solve'] = (df['last_activity'] - df['timestamp']).dt.total_seconds() / 3600 / 24

df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

if not show_total:
    filtered_df = df[
        df['type'].isin(selected_types) & df['district'].isin(selected_districts)
    ].copy()
else:
    filtered_df = df.copy()

filtered_df['timestamp_str'] = filtered_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M")
filtered_df['time2solve_str'] = filtered_df['time2solve'].round(2).astype(str)

# -------------------------------
# COLOR CONFIG
# -------------------------------
type_list = df['type'].unique()
cmap = cm.get_cmap('tab10', len(type_list))
color_dict_hex = {t: mcolors.to_hex(cmap(i)) for i, t in enumerate(type_list)}
color_dict_hex['รวมทั้งหมด'] = '#808080'

district_list = df['district'].unique()
district_map = cm.get_cmap('gist_ncar', len(district_list))
random.seed(42)
district_list = list(district_list)
random.shuffle(district_list)
district_colormap = {
    d: [int(c * 255) for c in district_map(i)[:3]] + [150]
    for i, d in enumerate(district_list)
}

filtered_df['color'] = filtered_df['district'].map(district_colormap)

# -------------------------------
# READY TO RENDER IN TABS
# -------------------------------



with tab1:
    st.header("📊 Overview: ภาพรวม Traffy Fondue")

    # --------- HEADER + INFO BOX ---------
    st.write("ข้อมูลนี้รวบรวมจาก Traffy Fondue เพื่อวิเคราะห์ปัญหาที่เกิดในแต่ละเขตของกรุงเทพมหานคร")
    st.info(f"🔍 ข้อมูลช่วงที่เลือก แสดงปัญหา **{filtered_df['type'].nunique()} ประเภท** จาก **{filtered_df['district'].nunique()} เขต**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 จำนวนปัญหาทั้งหมด", f"{filtered_df.shape[0]:,}")
    with col2:
        total_population = filtered_df[['district', 'population']].drop_duplicates()['population'].sum()
        st.metric("👥 ประชากรรวม", f"{total_population:,}")
    with col3:
        st.metric("📍 เขตที่เลือก", f"{filtered_df['district'].nunique()} เขต")

    top_district = filtered_df['district'].value_counts().reset_index()
    top_district.columns = ['district', 'case_count']
    if not top_district.empty:
        top1 = top_district.iloc[0]
        st.info(f"🚨 เขตที่มีปัญหารายงานมากที่สุด: **{top1['district']}** ({top1['case_count']:,} เคส)")
    else:
        st.warning("⚠️ กรุณาเลือกเขต")

    # --------- BAR + PIE CHART ---------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧱 จำนวนปัญหาแต่ละประเภท (แยกเขต)")
        show_total = st.checkbox("รวม 'ทั้งหมด' ต่อเขต", value=True)
        type_counts = filtered_df.groupby(['district', 'type']).size().reset_index(name='count')
        overall_counts = filtered_df.groupby('district').size().reset_index(name='count').assign(type='รวมทั้งหมด')
        full_counts = pd.concat([type_counts, overall_counts], ignore_index=True) if show_total else type_counts

        fig = px.bar(
            full_counts, x='district', y='count', color='type',
            color_discrete_map=color_dict_hex, barmode='group',
            text='count', labels={'count': 'จำนวน', 'district': 'เขต', 'type': 'ประเภท'}
        )
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🍰 สัดส่วนประเภทปัญหา")
        type_share = filtered_df['type'].value_counts().reset_index()
        type_share.columns = ['type', 'count']
        fig_pie = px.pie(type_share, names='type', values='count', color='type', color_discrete_map=color_dict_hex)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --------- MAP SECTION ---------
    st.subheader("🗺️ จุดที่เกิดปัญหาทั้งหมด")
        # วาด map plot
    st.header('Maping Plot')
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_df,
        get_position=["longitude", "latitude"],
        get_fill_color='color', 
        get_radius=20,
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=filtered_df['latitude'].mean(),
        longitude=filtered_df['longitude'].mean(),
        zoom=11,
    )
    tooltip = {
        "html": "<b>ประเภท:</b> {type}<br/><b>เขต:</b> {district}<br/><b>วันที่:</b> {timestamp_str}<br/>",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    if filtered_df.empty:
        st.warning("No data selected. Showing default point.")

    st.pydeck_chart(pdk.Deck(layers=[geojson_layer,layer], 
                            initial_view_state=view_state,
                            tooltip=tooltip,
                            ))
    # วาด map plot
    #บอกสี Map plot
    st.markdown("### สีของแต่ละเขต")
    with st.expander("ดูสีของแต่ละเขต"):
        for d, c in district_colormap.items():
            st.markdown(f"<span style='color: rgb({c[0]}, {c[1]}, {c[2]})'>⬤</span> {d}", unsafe_allow_html=True)

    #บอกสี Map plot

    # --------- DAILY CASE TREND ---------
    st.subheader("📆 แนวโน้มจำนวนเคสรายวัน")
    daily_case = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reset_index(name='count')
    fig = px.line(daily_case, x='timestamp', y='count', title='📈 จำนวนเคสที่รายงานในแต่ละวัน')
    st.plotly_chart(fig, use_container_width=True)

    # --------- TIME2SOLVE BY TYPE/DISTRICT ---------
    st.subheader("⏱️ เวลาเฉลี่ยในการจัดการปัญหา (รายเขต)")
    type_time2solve = filtered_df.groupby(['district', 'type'])['time2solve'].mean().reset_index(name='time2solve')
    overall_time2solve = filtered_df.groupby('district')['time2solve'].mean().reset_index(name='time2solve').assign(type='รวมทั้งหมด')
    full_time2solve = pd.concat([type_time2solve, overall_time2solve], ignore_index=True)

    fig_time2solve = px.bar(
        full_time2solve, x='district', y='time2solve', color='type',
        color_discrete_map=color_dict_hex, barmode='group',
        text=full_time2solve['time2solve'].round(1),
        labels={'time2solve': 'เวลาเฉลี่ย (วัน)', 'district': 'เขต', 'type': 'ประเภท'}
    )
    fig_time2solve.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_tickangle=-45)
    st.plotly_chart(fig_time2solve, use_container_width=True)

    # --------- TOP 5 FASTEST DISTRICTS ---------
    top_fast = filtered_df.groupby('district')['time2solve'].mean().sort_values().reset_index()
    # เรียงจำนวนวันจากน้อยไปมาก
    st.subheader("📋 ตารางจัดอันดับเขตตามเวลาเฉลี่ยในการจัดการปัญหา")

    if not top_fast.empty:
        fastest = top_fast.head(5).iloc[0]
        st.success(f"✅ เขตที่จัดการไวที่สุด: **{fastest['district']}** ({fastest['time2solve']:.2f} วัน)")
    fig = px.bar(top_fast.head(5), x='district', y='time2solve',
                title="🔥 5 เขตที่จัดการไวที่สุด (ค่าเฉลี่ยวัน)",
                labels={'time2solve': 'ระยะเวลา', 'district': 'เขต'},
                color='time2solve', color_continuous_scale='Greens_r')
    st.plotly_chart(fig, use_container_width=True)

    # Format ให้ดูสวย
    top_fast['time2solve'] = top_fast['time2solve'].round(2)
    top_fast.columns = ['เขต', 'เวลาเฉลี่ย (วัน)']

    # ใช้ dataframe UI ที่จัดหน้า
    st.dataframe(
        top_fast,
        use_container_width=True,
        hide_index=True
    )





with tab2:
    st.header("🏙️ District Performance")
    selected_district = st.selectbox("เลือกเขต", df['district'].unique() if not df.empty else [])

    if selected_district:
        district_df = df[df['district'] == selected_district].copy()

        st.markdown(f"### 📍 เขต: **{selected_district}**")
        st.caption("ข้อมูล performance รายเขต เช่น ความเร็วในการแก้ปัญหา งบ ฯลฯ")

        # ----- METRIC BOXES -----
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 จำนวนปัญหา", f"{district_df.shape[0]:,}")
        with col2:
            pop = district_df[['district', 'population']].drop_duplicates()['population'].values[0]
            st.metric("👥 ประชากร", f"{pop:,}")
        with col3:
            budget = district_df['budget'].mean()
            st.metric("💰 งบเฉลี่ยต่อเคส", f"{budget:,.2f} บาท")
        avg_duration = district_df['time2solve'].mean()
        st.metric("⏱️ เวลาเฉลี่ยในการแก้ปัญหา", f"{avg_duration:.2f} วัน")

        # ----- PIE CHART: TYPE SHARE -----
        st.subheader("📊 สัดส่วนประเภทปัญหาในเขตนี้")
        type_share = district_df['type'].value_counts().reset_index()
        type_share.columns = ['type', 'count']
        fig_pie = px.pie(
            type_share, names='type', values='count',
            color='type', color_discrete_map=color_dict_hex,
            title=f"🧾 สัดส่วนประเภทปัญหาในเขต {selected_district}"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("📄 เคสล่าสุดในเขตนี้")
        latest_cases = district_df[['timestamp', 'type', 'comment', 'state']].sort_values(by='timestamp', ascending=False).head(10)
        latest_cases['timestamp'] = latest_cases['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(latest_cases, use_container_width=True, hide_index=True)

        # ----- BAR CHART: TIME2SOLVE PER TYPE -----
        st.subheader("⏱️ ระยะเวลาเฉลี่ยต่อประเภทปัญหา")
        type_duration = district_df.groupby('type')['time2solve'].mean().reset_index()
        fig = px.bar(
            type_duration, x='type', y='time2solve',
            color='type', color_discrete_map=color_dict_hex,
            title=f"🛠️ ระยะเวลาเฉลี่ยในการจัดการแต่ละประเภท ({selected_district})",
            labels={'time2solve': 'เวลาเฉลี่ย (วัน)', 'type': 'ประเภท'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # ----- TIME SERIES: OPEN VS CLOSED VS BACKLOG -----
        st.subheader("📆 จำนวนปัญหาที่รับ vs ปิดในแต่ละวัน")
        district_df['date_opened'] = pd.to_datetime(district_df['timestamp']).dt.date
        district_df['date_closed'] = pd.to_datetime(district_df['last_activity']).dt.date
        open_daily = district_df.groupby('date_opened').size().reset_index(name='opened')
        closed_daily = district_df.groupby('date_closed').size().reset_index(name='closed')

        open_close_df = pd.merge(
            open_daily, closed_daily,
            left_on='date_opened', right_on='date_closed',
            how='outer'
        )
        open_close_df['date'] = open_close_df['date_opened'].combine_first(open_close_df['date_closed'])
        open_close_df['opened'] = open_close_df['opened'].fillna(0)
        open_close_df['closed'] = open_close_df['closed'].fillna(0)
        open_close_df = open_close_df.sort_values('date')
        open_close_df['backlog'] = open_close_df['opened'].cumsum() - open_close_df['closed'].cumsum()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=open_close_df['date'], y=open_close_df['opened'],
                                      mode='lines+markers', name='รับเรื่อง'))
        fig_line.add_trace(go.Scatter(x=open_close_df['date'], y=open_close_df['closed'],
                                      mode='lines+markers', name='จัดการเสร็จ'))
        fig_line.add_trace(go.Scatter(x=open_close_df['date'], y=open_close_df['backlog'],
                                      mode='lines', name='Backlog สะสม',
                                      line=dict(dash='dash')))
        fig_line.update_layout(
            title="📈 เปรียบเทียบจำนวนปัญหาที่รับเรื่อง vs จัดการเสร็จ (รายวัน)",
            xaxis_title="วันที่", yaxis_title="จำนวนเคส", legend_title="สถานะ"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("📌 ถ้าเส้น 'จัดการเสร็จ' สูงกว่า 'รับเรื่อง' แปลว่าเป็นการเคลียร์ของที่ดองไว้ก่อนหน้า")





df3 = results.copy()

with tab3:
    st.header("📈 Model Results")
    st.write("ผลลัพธ์จากโมเดลที่เทรน เช่น เขตไหนจัดการดีสุด ประสิทธิภาพ ฯลฯ")
    st.subheader("🏆 เขตที่มีประสิทธิภาพมากที่สุดในการจัดการปัญหา")

    # คำนวณค่าเฉลี่ย time2solve ของแต่ละเขต
    eff_df = df3.groupby('district')['time2solve'].mean().reset_index().sort_values('time2solve')

    # ดึงเขตที่เร็วที่สุด
    if not eff_df.empty:
        best = eff_df.iloc[0]
        st.success(f"🥇 เขตที่จัดการปัญหาเร็วที่สุดคือ **{best['district']}** เฉลี่ยเพียง **{best['time2solve']:.2f} วัน**")

    # วาดกราฟ
    fig = px.bar(
        eff_df.head(10),  # Top 10 ก็พอ
        x='district',
        y='time2solve',
        title="🔥 10 เขตที่ใช้เวลาเฉลี่ยน้อยที่สุดในการจัดการปัญหา",
        labels={'district': 'เขต', 'time2solve': 'เวลาเฉลี่ย (วัน)'},
        color='time2solve',
        color_continuous_scale='Tealgrn_r',
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    slowest_df = eff_df.sort_values('time2solve', ascending=False).head(10)
    # เขตที่จัดการช้าที่สุด
    worst = slowest_df.iloc[0]
    st.error(f"🐌 เขตที่จัดการปัญหาช้าที่สุดคือ **{worst['district']}** เฉลี่ย **{worst['time2solve']:.2f} วัน**")
    st.markdown("### 🔴 เขตที่ใช้เวลานานที่สุด (TOP 10)")
    fig_slow = px.bar(
        slowest_df,
        x='district',
        y='time2solve',
        labels={'district': 'เขต', 'time2solve': 'เวลาเฉลี่ย (วัน)'},
        title="🐌 เขตที่ใช้เวลานานที่สุด",
        color='time2solve',
        color_continuous_scale='Reds'
    )
    fig_slow.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_slow, use_container_width=True)
    

    st.subheader("🤖 Time2Solve Prediction Performance")
    st.caption("เปรียบเทียบผลลัพธ์จากโมเดลกับค่าจริง เพื่อประเมินความแม่นยำของระบบที่พัฒนา")
    
    # MAE, RMSE, R²
    mae_rf = mean_absolute_error(df3['time2solve'], df3['time2solve_BERT_rf_pred'])
    mae_xgb = mean_absolute_error(df3['time2solve'], df3['time2solve_BERT_xgb_pred'])
    rmse_rf = np.sqrt(mean_squared_error(df3['time2solve'], df3['time2solve_BERT_rf_pred']))
    rmse_xgb = np.sqrt(mean_squared_error(df3['time2solve'], df3['time2solve_BERT_xgb_pred']))
    r2_rf = r2_score(df3['time2solve'], df3['time2solve_BERT_rf_pred'])
    r2_xgb = r2_score(df3['time2solve'], df3['time2solve_BERT_xgb_pred'])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📌 MAE (RF)", f"{mae_rf:.2f}")
        st.metric("📌 RMSE (RF)", f"{rmse_rf:.2f}")
        st.metric("📌 R² (RF)", f"{r2_rf:.2f}")
    with col2:
        st.metric("📌 MAE (XGB)", f"{mae_xgb:.2f}")
        st.metric("📌 RMSE (XGB)", f"{rmse_xgb:.2f}")
        st.metric("📌 R² (XGB)", f"{r2_xgb:.2f}")

    # -------- MAE by District --------
    st.subheader("🏙️ เปรียบเทียบ MAE ของ RF vs XGB (รายเขต)")
    district_mae = df3.groupby('district').apply(lambda g: pd.Series({
        'MAE_RF': mean_absolute_error(g['time2solve'], g['time2solve_BERT_rf_pred']),
        'MAE_XGB': mean_absolute_error(g['time2solve'], g['time2solve_BERT_xgb_pred']),
    })).reset_index()
    fig = px.bar(district_mae, x='district', y=['MAE_RF', 'MAE_XGB'],
                 barmode='group',
                 labels={'value': 'MAE', 'district': 'เขต', 'variable': 'โมเดล'})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(district_mae, use_container_width=True)

    # -------- MAE by Type --------
    st.subheader("🧾 เปรียบเทียบ MAE ของ RF vs XGB (รายประเภทปัญหา)")
    type_mae = df3.groupby('type').apply(lambda g: pd.Series({
        'MAE_RF': mean_absolute_error(g['time2solve'], g['time2solve_BERT_rf_pred']),
        'MAE_XGB': mean_absolute_error(g['time2solve'], g['time2solve_BERT_xgb_pred']),
    })).reset_index()

    fig = px.bar(
        type_mae,
        x='type',
        y=['MAE_RF', 'MAE_XGB'],
        barmode='group',
        labels={'value': 'MAE', 'type': 'ประเภทปัญหา', 'variable': 'โมเดล'},
        title="📊 เปรียบเทียบ MAE ของ RF vs XGB (รายประเภทปัญหา)"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 MAE แยกตามประเภทปัญหา")
    type_mae = df3.groupby('type').apply(lambda g: pd.Series({
        'MAE_RF': mean_absolute_error(g['time2solve'], g['time2solve_BERT_rf_pred']),
        'MAE_XGB': mean_absolute_error(g['time2solve'], g['time2solve_BERT_xgb_pred']),
    })).reset_index()
    st.dataframe(type_mae, use_container_width=True)
