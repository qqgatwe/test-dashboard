import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import jieba
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud


# =========================================================
# 1. 頁面配置
# =========================================================

st.set_page_config(
    page_title="LCCNET 文章爬蟲與文字分析",
    layout="wide"
)

st.title("📊 LCCNET 文章爬蟲與文字分析")

st.caption(
    "使用 Requests、BeautifulSoup、re、Jieba、Pandas "
    "與 Matplotlib 分析真實網站文章"
)


# =========================================================
# 2. 網站設定
# =========================================================

URL = "https://www.lccnet.com.tw/lccnet/article/details/2733"


# =========================================================
# 3. Requests 取得網站
# =========================================================

@st.cache_data(ttl=1800)
def get_web_page():

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        URL,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    response.encoding = response.apparent_encoding

    return response.text


# =========================================================
# 4. BeautifulSoup 解析文章
# =========================================================

def parse_article(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # -----------------------------------------------------
    # 網頁標題
    # -----------------------------------------------------

    page_title = "未知標題"

    if soup.title:

        page_title = soup.title.get_text(
            strip=True
        )


    # -----------------------------------------------------
    # 找真正的文章 H1
    # -----------------------------------------------------

    article_title = None

    for h1 in soup.find_all("h1"):

        text = h1.get_text(
            " ",
            strip=True
        )

        if "非本科轉職工程師" in text:

            article_title = h1
            break


    if article_title is None:

        article_title = soup.find("h1")


    # -----------------------------------------------------
    # 儲存文章內容
    # -----------------------------------------------------

    article_items = []


    if article_title:

        title_text = article_title.get_text(
            " ",
            strip=True
        )

        if title_text:

            article_items.append({
                "type": "h1",
                "text": title_text
            })


        # -------------------------------------------------
        # 往後找文章內容
        # -------------------------------------------------

        current = article_title.find_next()

        while current:

            tag_name = current.name


            # 遇到 Footer / iframe 停止
            if tag_name in [
                "footer",
                "iframe"
            ]:

                break


            # -------------------------------------------------
            # H2
            # -------------------------------------------------

            if tag_name == "h2":

                text = current.get_text(
                    " ",
                    strip=True
                )

                if text:

                    article_items.append({
                        "type": "h2",
                        "text": text
                    })


            # -------------------------------------------------
            # H3
            # -------------------------------------------------

            elif tag_name == "h3":

                text = current.get_text(
                    " ",
                    strip=True
                )

                if text:

                    article_items.append({
                        "type": "h3",
                        "text": text
                    })


            # -------------------------------------------------
            # H4
            # -------------------------------------------------

            elif tag_name == "h4":

                text = current.get_text(
                    " ",
                    strip=True
                )

                if text:

                    article_items.append({
                        "type": "h4",
                        "text": text
                    })


            # -------------------------------------------------
            # P
            # -------------------------------------------------

            elif tag_name == "p":

                text = current.get_text(
                    " ",
                    strip=True
                )

                if text:

                    article_items.append({
                        "type": "p",
                        "text": text
                    })


            # -------------------------------------------------
            # LI
            # -------------------------------------------------

            elif tag_name == "li":

                text = current.get_text(
                    " ",
                    strip=True
                )

                if text:

                    article_items.append({
                        "type": "li",
                        "text": text
                    })


            current = current.find_next()


    # =====================================================
    # 排除目錄
    # =====================================================

    cleaned_items = []

    skip_toc = False

    for item in article_items:

        text = item["text"]


        if text == "目錄":

            skip_toc = True
            continue


        if skip_toc and item["type"] == "h2":

            skip_toc = False


        if not skip_toc:

            cleaned_items.append(item)


    # =====================================================
    # 移除重複
    # =====================================================

    final_items = []

    seen = set()

    for item in cleaned_items:

        key = (
            item["type"],
            item["text"]
        )

        if key not in seen:

            final_items.append(item)

            seen.add(key)


    # =====================================================
    # 組合文章文字
    # =====================================================

    article_text = "\n".join(
        item["text"]
        for item in final_items
    )


    return (
        page_title,
        article_text,
        final_items
    )


# =========================================================
# 5. 文字清理
# =========================================================

def clean_text(text):

    # 移除網址
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # 移除 HTML
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # 保留中文、英文、數字
    text = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9]",
        " ",
        text
    )

    # 多個空白
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# 6. Jieba 中文斷詞
# =========================================================

def word_segmentation(text):

    words = jieba.lcut(text)


    # -----------------------------------------------------
    # 停用詞
    # -----------------------------------------------------

    stop_words = {

        "的", "了", "是", "在",
        "有", "和", "與", "也",
        "就", "都", "而", "及",
        "或", "一", "不",

        "可以", "如果",
        "這", "那", "將",
        "讓", "為", "從",

        "以及", "我們",
        "你", "我", "他",
        "她", "它",

        "這些", "這個",

        "如何", "因此",
        "透過", "開始",
        "進行",

        "比較", "可能",
        "目前", "一個",

        "就是", "自己",
        "沒有", "還有",

        "不是", "因為",
        "所以",

        "其中", "這篇",
        "文章",

        "相關", "內容",
        "方式", "部分",
        "問題", "地方"
    }


    result = []


    for word in words:

        word = word.strip()


        # 空白
        if not word:
            continue


        # 停用詞
        if word in stop_words:
            continue


        # 一個字
        if len(word) < 2:
            continue


        # 純數字
        if word.isdigit():
            continue


        result.append(word)


    return result


# =========================================================
# 7. Matplotlib 中文字體
# =========================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Noto Sans CJK TC"
]

plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 8. 抓取網站
# =========================================================

try:

    html = get_web_page()

except requests.exceptions.RequestException as e:

    st.error(
        f"❌ 網站資料取得失敗：{e}"
    )

    st.stop()


# =========================================================
# 9. 解析文章
# =========================================================

try:

    page_title, article_text, article_items = parse_article(
        html
    )

except Exception as e:

    st.error(
        f"❌ 網頁解析失敗：{e}"
    )

    st.stop()


# =========================================================
# 10. 文字清理
# =========================================================

cleaned_text = clean_text(
    article_text
)


# =========================================================
# 11. Jieba 斷詞
# =========================================================

words = word_segmentation(
    cleaned_text
)


# =========================================================
# 12. 詞頻統計
# =========================================================

word_count = Counter(
    words
)


# =========================================================
# 13. DataFrame
# =========================================================

wordcloud_df = pd.DataFrame(
    word_count.items(),
    columns=[
        "word",
        "count"
    ]
)

wordcloud_df = wordcloud_df.sort_values(
    "count",
    ascending=False
).reset_index(
    drop=True
)


# =========================================================
# 14. 頁面資訊
# =========================================================

st.subheader("📄 文章資訊")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "文章內容區塊",
        len(article_items)
    )


with col2:

    st.metric(
        "有效詞彙數",
        len(words)
    )


with col3:

    st.metric(
        "不同詞彙數",
        len(word_count)
    )


st.write(
    "**文章標題：**",
    page_title
)

st.write(
    "**資料來源：**",
    URL
)


# =========================================================
# 15. Tabs
# =========================================================

tab_article, tab_analysis, tab_wordcloud = st.tabs(
    [
        "📄 文章內容",
        "📊 關鍵字分析",
        "☁️ 文字雲"
    ]
)


# =========================================================
# 【Tab 1】文章內容
# =========================================================

with tab_article:

    st.subheader(
        "📄 LCCNET 文章內容"
    )


    with st.expander(
        "📖 查看完整文章內容"
    ):

        for item in article_items:

            if item["type"] == "h1":

                st.title(
                    item["text"]
                )

            elif item["type"] == "h2":

                st.subheader(
                    item["text"]
                )

            elif item["type"] == "h3":

                st.markdown(
                    f"### {item['text']}"
                )

            elif item["type"] == "h4":

                st.markdown(
                    f"#### {item['text']}"
                )

            elif item["type"] == "li":

                st.markdown(
                    f"- {item['text']}"
                )

            else:

                st.write(
                    item["text"]
                )


# =========================================================
# 【Tab 2】關鍵字分析
# =========================================================

with tab_analysis:

    st.subheader(
        "💬 文章熱門關鍵字分析"
    )

    st.write(
        "透過 Jieba 中文斷詞，統計文章中出現頻率較高的詞彙。"
    )


    # -----------------------------------------------------
    # Top 20
    # -----------------------------------------------------

    top_words = wordcloud_df.head(
        20
    )


    # -----------------------------------------------------
    # DataFrame
    # -----------------------------------------------------

    st.dataframe(
        top_words,
        width="stretch"
    )


    # -----------------------------------------------------
    # Matplotlib 長條圖
    # -----------------------------------------------------

    st.subheader(
        "📊 Top 20 Keyword Frequency"
    )


    plot_df = top_words.sort_values(
        "count",
        ascending=True
    )


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    ax.barh(
        plot_df["word"],
        plot_df["count"]
    )


    ax.set_title(
        "LCCNET Article Keyword Frequency",
        fontsize=16
    )


    ax.set_xlabel(
        "Frequency"
    )


    ax.set_ylabel(
        "Keyword"
    )


    plt.tight_layout()


    st.pyplot(
        fig,
        width="stretch"
    )


# =========================================================
# 【Tab 3】文字雲
# =========================================================

with tab_wordcloud:

    st.subheader(
        "☁️ LCCNET 文章文字雲"
    )

    st.write(
        "字體大小代表詞彙在文章中的出現頻率。"
    )


    # =====================================================
    # WordCloud 中文字體
    # =====================================================

    font_path = r"C:\Windows\Fonts\msjh.ttc"


    # =====================================================
    # 建立文字雲
    # =====================================================

    wc = WordCloud(
        font_path=font_path,
        width=1200,
        height=700,
        background_color="white",
        max_words=100,
        min_font_size=12,
        max_font_size=100,
        collocations=False,
        random_state=42
    )


    # =====================================================
    # 使用詞頻資料產生文字雲
    # =====================================================

    wc.generate_from_frequencies(
        word_count
    )


    # =====================================================
    # Matplotlib 顯示文字雲
    # =====================================================

    fig_wc, ax_wc = plt.subplots(
        figsize=(14, 8)
    )


    ax_wc.imshow(
        wc,
        interpolation="bilinear"
    )


    ax_wc.axis(
        "off"
    )


    ax_wc.set_title(
        "LCCNET Article Word Cloud",
        fontsize=20,
        pad=20
    )


    plt.tight_layout()


    st.pyplot(
        fig_wc,
        width="stretch"
    )


    # =====================================================
    # 顯示 Top 10
    # =====================================================

    st.subheader(
        "🏆 文字雲主要關鍵字"
    )


    top10 = wordcloud_df.head(
        10
    )


    for index, row in top10.iterrows():

        st.write(
            f"**{index + 1}. {row['word']}** "
            f"— 出現 {row['count']} 次"
        )


# =========================================================
# 16. CSV 下載
# =========================================================

st.divider()

st.subheader(
    "📥 資料下載"
)


csv = wordcloud_df.to_csv(
    index=False,
    encoding="utf-8-sig"
)


st.download_button(
    label="下載關鍵字分析 CSV",
    data=csv,
    file_name="lccnet_keyword_analysis.csv",
    mime="text/csv"
)