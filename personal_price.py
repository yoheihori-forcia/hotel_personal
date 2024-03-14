import streamlit as st
import pandas as pd
import numpy as np 
import collections

def initialze_page():
    st.set_page_config(
        page_title = "Hotel Recommend Tools",
        page_icon = "🏨"
    )

def cos_similarity(a,b):
    return np.dot(a,b) / ((np.sqrt(np.dot(a,a))) * (np.sqrt(np.dot(b,b))))

def df_to_lists(display:pd.DataFrame):
    urls = display['url'].to_list()
    titles = display['title'].to_list()
    contents =display['content'].to_list()
    names = display['name'].to_list()
    embeddings = display['embedding'].to_list()
    idos = display['ido'].to_list()
    keidos = display['keido'].to_list()
    hotelids = display['hotelid'].to_list()
    prices = display['price'].to_list()

    return urls,titles,contents,names,embeddings,idos,keidos,hotelids,prices


def add_vector():
    st.session_state['personal_v'] = [x+y for x,y in zip(st.session_state['personal_v'],st.session_state['embedding'])]

def personalize(df:pd.DataFrame,history_df:pd.DataFrame):
    history_p = history_df['hotelid'].to_list()
    ids = df['hotelid'].to_list()
    embeddings = df['embedding'].to_list()
    personal_v= np.zeros(1024).tolist()
    pref = []
    for id in history_p:
        if id in ids:
            personal_v = [x+y for x,y in zip(personal_v,embeddings[ids.index(id)])]
            pref.append(id[1:3])
    pref_most = collections.Counter(pref).most_common()

    return personal_v, pref_most, history_df

def add_vector():
    st.session_state['personal_v'] = [x+y for x,y in zip(st.session_state['personal_v'],st.session_state['embedding'])]


def set_price_range(df:pd.DataFrame,min:int,max:int):
    if min < 20000:
        minp = min*0.25
    elif min < 50000:
        minp = min*5/6 - (50000/3 - 5000)
    elif min < 100000:
        minp = min*7/10 - 5000
    else:
        minp = 65000

    maxp = 2*max + 20000

    df = df[df['price'] <= maxp]
    df = df[df['price'] >= minp]

    return df

def main():
    st.header("宿泊履歴に基づくおすすめ")
    search = st.text_input("宿泊施設を名称で検索")
    if 'hist_df' not in st.session_state:
        st.session_state['hist_df'] = pd.DataFrame()
        st.session_state['hist_name'] = []
        st.session_state['hist_price'] = []
        st.session_state['hist_url'] = []
        st.session_state['hist_id'] = []
        st.session_state['personal_v'] = np.zeros(1024)

    def load_vdb():
        return pd.read_pickle('vector_database.pkl')
    df = load_vdb()
    st.session_state['df'] = df
    search_vdb = df[df['name'].str.contains(search)]

    def add_history(i):
        namelist = st.session_state['hist_name']
        namelist.append(sname[i])
        st.session_state['hist_name'] = namelist

        pricelist = st.session_state['hist_price']
        pricelist.append(sprice[i])
        st.session_state['hist_price'] = pricelist

        urllist = st.session_state['hist_url']
        urllist.append(surl[i])
        st.session_state['hist_url'] = urllist

        idlist = st.session_state['hist_id']
        idlist.append(shotelid[i])
        st.session_state['hist_id'] = idlist

        df = pd.concat([pd.Series(st.session_state['hist_name']),pd.Series(st.session_state['hist_price']),pd.Series(st.session_state['hist_url']),pd.Series(st.session_state['hist_id'])], axis=1)
        df.columns = ['name','price','url', 'hotelid']
        st.session_state['hist_df'] = df.reset_index(drop=True)


    surl,stitle,scontent,sname,sembedding,sido,skeido,shotelid,sprice = df_to_lists(search_vdb)
    limit = 0

    for i in range(len(search_vdb)):
        if limit > 19:
            break
        limit += 1
        dname, durl, dprice = sname[i], surl[i], sprice[i]
        st.markdown(f'**{dname}**  {dprice}円～  {durl}')
        st.button(f"{dname}を履歴に追加", on_click=add_history,args=(i,))



    def del_history(i):
        namelist = st.session_state['hist_name']
        namelist.remove(gname[i])
        st.session_state['hist_name'] = namelist

        pricelist = st.session_state['hist_price']
        pricelist.remove(gprice[i])
        st.session_state['hist_price'] = pricelist

        urllist = st.session_state['hist_url']
        urllist.remove(gurl[i])
        st.session_state['hist_url'] = urllist

        idlist = st.session_state['hist_id']
        idlist.remove(ghotelid[i])
        st.session_state['hist_id'] = idlist
        df = pd.concat([pd.Series(st.session_state['hist_name']),pd.Series(st.session_state['hist_price']),pd.Series(st.session_state['hist_url']),pd.Series(st.session_state['hist_id'])], axis=1)
        df.columns = ['name','price','url','hotelid']
        st.session_state['hist_df'] = df.reset_index(drop=True)
    
    
    if len(st.session_state.hist_df) !=0:
        st.subheader("現在の履歴")
        gname = st.session_state.hist_df['name'].to_list()
        gurl = st.session_state.hist_df['url'].to_list()
        gprice = st.session_state.hist_df['price'].to_list()
        ghotelid = st.session_state.hist_df['hotelid'].to_list()
        for i in range(len(gname)):
            genname, genurl, genprice = gname[i], gurl[i], gprice[i]
            st.markdown(f'**{genname}**  {genprice}円～  {genurl}')
            st.button(f"{genname}を履歴から削除 {i}", on_click=del_history, args=(i,))
        st.session_state.personal_v = np.zeros(1024)
        for i in range(len(st.session_state['hist_name'])):
            st.session_state.personal_v = st.session_state.personal_v + np.array(st.session_state.df['embedding'].to_list()[st.session_state.df['name'].to_list().index(st.session_state['hist_name'][i])]) 


        st.header("上記の履歴を基にした、あなたへのおすすめホテル")
        sim = []
        embeddings = set_price_range(st.session_state.df, min(st.session_state['hist_price']), max(st.session_state['hist_price']))['embedding'].to_list()
        for embedding in embeddings:
            sim.append(cos_similarity(st.session_state['personal_v'], embedding))
        personal = set_price_range(st.session_state.df, min(st.session_state['hist_price']), max(st.session_state['hist_price']))
        personal['sim'] = sim
        personal = personal.sort_values('gacount', ascending=False).head(int(len(df)*0.2))
        personal = personal.sort_values('sim', ascending=False)
        personal = personal.iloc[0:10].reset_index(drop=True)
        personal = personal.sort_values("gacount", ascending=False).reset_index(drop=True)


        urlsp,titlesp,contentsp,namesp,embeddingsp,idosp,keidosp,hotelidsp,pricesp = df_to_lists(personal)

        for i in range(len(personal)):
            st.session_state["url"] = urlsp[i]
            st.session_state["title"] = titlesp[i]
            st.session_state["content"] = contentsp[i]
            name, price, url = namesp[i], pricesp[i], urlsp[i]
            st.markdown(f'**{name}**  \n{price}円～ {url}')
 

def detail():
    st.headaer("工事中")

pages = dict(
    page1="おすすめ",
    page2="詳細",
)

page_id = st.sidebar.selectbox(
    "ページ名",
    ["page1", "page2"],
    format_func=lambda page_id: pages[page_id],
    key = "page-select",
)

if page_id == "page1":
    main()

if page_id == "page2":
    detail()