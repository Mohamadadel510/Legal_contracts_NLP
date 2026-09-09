# LegalLens - Build RAG Index
# Input: data/laws.jsonl
# Output: data/legal_articles.json, legal_articles.csv,
#         article_metadata.json, embeddings.npy, bm25.pkl,
#         index/legal_rag_qdrant/

import argparse, json, pickle, re, csv
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

SUPPORTED = {"residential","agricultural","commercial","employment","company","sale","supply","service","consumer"}

KEYWORDS = {
"agricultural":["الأرض الزراعية","الارض الزراعية","الأراضي الزراعية","الاراضي الزراعية","إيجار الأرض الزراعية","ايجار الارض الزراعية","المزارعة","الدورة الزراعية","المحصول","استغلال الأرض الزراعية","استغلال الارض الزراعية","مواشي وأدوات زراعية","مواشي وادوات زراعية","أدوات زراعية","ادوات زراعية","الأرض صالحة للإنتاج","الارض صالحة للانتاج"],
"residential":["السكن","السكنى","المسكن","المساكن","الشقة","الشقق","العين المؤجرة للسكن","المكان المعد للسكن"],
"employment":["عقد العمل","عقد عمل","علاقة العمل","صاحب العمل","العامل","العمال","الأجر المستحق للعامل","إنهاء عقد العمل"],
"commercial":["المحل التجاري","المحلات التجارية","الأعمال التجارية","التاجر","التجار","المتجر","الدفاتر التجارية","السجل التجاري"],
"company":["شركة مساهمة","شركة التضامن","شركة التوصية","شركة ذات مسئولية محدودة","شركة ذات مسؤولية محدودة","الشريك","الشركاء","رأس مال الشركة","راس مال الشركة"],
"sale":["عقد البيع","عقد بيع","البائع","المشتري","المبيع","الثمن"],
"supply":["عقد التوريد","عقد توريد","التوريد","المورد"],
"service":["عقد تقديم الخدمات","عقد تقديم خدمة","تقديم الخدمات","مقدم الخدمة"],
"consumer":["حماية المستهلك","قانون حماية المستهلك","المستهلك","حقوق المستهلك","واجبات المورد"]}

def norm(s):
    s = str(s or "").lower()
    s = re.sub(r"[\u064B-\u065F\u0670]", "", s)
    for a,b in [("أ","ا"),("إ","ا"),("آ","ا"),("ٱ","ا"),("ى","ي")]: s=s.replace(a,b)
    return re.sub(r"\s+", " ", s.replace("ـ", "")).strip()

NK = {k:[norm(x) for x in v] for k,v in KEYWORDS.items()}

def tokenize(s): return re.findall(r"[\u0600-\u06FF]+", norm(s))

def classify(a):
    existing = a.get("contract_types")
    if isinstance(existing,str): existing=[existing]
    if isinstance(existing,list) and existing:
        valid=[x for x in existing if x in SUPPORTED or x=="general"]
        if valid:
            return list(dict.fromkeys(valid+["general"]))
    text=norm(a.get("text",""))
    found=[k for k,words in NK.items() if any(w and w in text for w in words)]
    return list(dict.fromkeys(found+["general"]))

def load_jsonl(path):
    out=[]
    with open(path,encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: a=json.loads(line)
            except Exception as e: raise ValueError(f"Invalid JSON at line {n}: {e}")
            if not isinstance(a,dict) or not str(a.get("text","")).strip(): raise ValueError(f"Line {n} must contain a non-empty 'text'")
            out.append(a)
    if not out: raise ValueError("No articles found")
    return out

def normalize_article(a):
    x={
        "law_id":a.get("law_id"),"law_name":a.get("law_name"),"law_number":a.get("law_number"),"law_year":a.get("law_year"),
        "article_number":a.get("article_number"),"article_label":a.get("article_label",f"مادة {a.get('article_number')}" if a.get("article_number") is not None else None),
        "chapter":a.get("chapter"),"binding_type":a.get("binding_type","general"),
        "needs_executive_regulation":a.get("needs_executive_regulation",False),"superseded_by":a.get("superseded_by"),
        "is_active":a.get("is_active",True),"start_page":a.get("start_page"),"end_page":a.get("end_page"),"text":str(a.get("text","" )).strip()}
    x["contract_types"]=classify(a)
    return x

def save_articles(articles,data):
    data.mkdir(parents=True,exist_ok=True)
    with open(data/"legal_articles.json","w",encoding="utf-8") as f: json.dump(articles,f,ensure_ascii=False,indent=2)
    fields=list(articles[0].keys())
    with open(data/"legal_articles.csv","w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for a in articles:
            r=dict(a); r["contract_types"]=json.dumps(r["contract_types"],ensure_ascii=False); w.writerow(r)
    meta=[{k:a.get(k) for k in fields if k!="text"} for a in articles]
    with open(data/"article_metadata.json","w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)

def build_bm25(articles,data):
    corpus=[tokenize(a["text"]) for a in articles]
    bm25=BM25Okapi(corpus)
    with open(data/"bm25.pkl","wb") as f: pickle.dump({"bm25":bm25,"corpus":corpus},f)

def build_embeddings(articles,data,model_name):
    model=SentenceTransformer(model_name)
    emb=model.encode([a["text"] for a in articles],normalize_embeddings=True,show_progress_bar=True,batch_size=32)
    emb=np.asarray(emb,dtype=np.float32); np.save(data/"embeddings.npy",emb); return emb

def build_qdrant(articles,emb,index_dir,collection):
    path=index_dir/"legal_rag_qdrant"; index_dir.mkdir(parents=True,exist_ok=True)
    q=QdrantClient(path=str(path))
    if q.collection_exists(collection): q.delete_collection(collection)
    q.create_collection(collection_name=collection,vectors_config=VectorParams(size=emb.shape[1],distance=Distance.COSINE))
    for start in range(0,len(articles),100):
        pts=[PointStruct(id=i,vector=emb[i].tolist(),payload=a) for i,a in enumerate(articles[start:start+100],start=start)]
        q.upsert(collection_name=collection,points=pts,wait=True); print(f"Uploaded {min(start+100,len(articles))}/{len(articles)}")
    count=q.count(collection_name=collection).count; q.close()
    if count!=len(articles): raise RuntimeError(f"Qdrant count mismatch: {count} != {len(articles)}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/laws.jsonl")
    p.add_argument("--data-dir",default="data")
    p.add_argument("--index-dir",default="index")
    p.add_argument("--model",default="BAAI/bge-m3")
    p.add_argument("--collection",default="egyptian_legal_articles_contract_types")
    args=p.parse_args()
    data=Path(args.data_dir); index=Path(args.index_dir)
    raw=load_jsonl(Path(args.input)); articles=[normalize_article(a) for a in raw]
    print(f"Loaded: {len(articles)} articles")
    dist={}
    for a in articles:
        for t in a["contract_types"]: dist[t]=dist.get(t,0)+1
    print("Contract types:",dist)
    save_articles(articles,data); build_bm25(articles,data); emb=build_embeddings(articles,data,args.model)
    if len(articles)!=len(emb): raise RuntimeError("Articles/embeddings mismatch")
    build_qdrant(articles,emb,index,args.collection)
    print("\nBUILD COMPLETED SUCCESSFULLY")
    print(f"Articles: {len(articles)} | Embeddings: {emb.shape}")
    print(f"Qdrant: {index/'legal_rag_qdrant'}")

if __name__=="__main__": main()
