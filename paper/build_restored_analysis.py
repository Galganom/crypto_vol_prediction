"""Build original-format paper tables and restored empirical analyses.

Primary benchmark arrays are read unchanged. New model fits are read from
results/ablations/paper_revision; new post-hoc labels and analyses are kept in
results/analysis/paper_revision. No unobserved experiment values are filled in.
"""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR","/tmp/crypto-paper-matplotlib")
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, mannwhitneyu, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from main_model_common import compute_metrics
from ablations.zscore_decile_improvement import forward_4h_log_return

PAPER=ROOT/"paper"
GEN=PAPER/"generated"
FIG=PAPER/"figures"
ANALYSIS=ROOT/"results/analysis/paper_revision"
ABLATIONS=ROOT/"results/ablations/paper_revision"
ANALYSIS.mkdir(parents=True,exist_ok=True)
MODELS=[("har","HAR"),("egarch","EGARCH"),("egarch_ned","EGARCH+NED"),("ts_mlp","TS-only"),("main_model","NED")]


def tex(text):
    trans={"&":r"\&","%":r"\%","$":r"\$","#":r"\#","_":r"\_","{":r"\{","}":r"\}"}
    return "".join(trans.get(c,c) for c in str(text)).replace("–","--").replace("—","---").replace("’","'")


def table(name,columns,header,rows):
    s=[r"\begin{tabular}{"+columns+"}",r"\toprule",header+r" \\",r"\midrule"]
    s += [row if isinstance(row,str) else " & ".join(map(str,row))+r" \\" for row in rows]
    s += [r"\bottomrule",r"\end{tabular}"]
    (GEN/name).write_text("\n".join(s)+"\n")


def hln(first,second,h):
    d=np.asarray(first,np.float64)-np.asarray(second,np.float64)
    n=len(d);center=d-d.mean();omega=np.mean(center**2)
    # Original manuscript protocol: Bartlett HAC through h-1 lags.
    for j in range(1,h):omega+=2*(1-j/h)*np.mean(center[j:]*center[:-j])
    stat=d.mean()/np.sqrt(omega/n)*np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    return float(stat),float(2*student_t.sf(abs(stat),n-1))


def stars(p):
    s="***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
    return "$^{"+s+"}$" if s else ""


def percentage(value):
    # Keep very small effects visible rather than printing a signed zero.
    return f"{value:+.3f}" if 0<abs(value)<.05 else f"{value:+.1f}"


def main_tables():
    rows=[];records={};dms={};means={}
    for asset in ("BTC","ETH"):
        base=ROOT/f"results/main/{asset.lower()}_cryptobert"
        for horizon,h in (("1h",4),("4h",16)):
            loss_store={name:{"MSE":[],"QLIKE":[]} for _,name in MODELS}
            for window in range(1,7):
                data=np.load(base/f"predictions/window_{window}_{horizon}_predictions.npz")
                ms={name:compute_metrics(data[key].astype(np.float64),data["actual"]) for key,name in MODELS}
                for name,m in ms.items():
                    records[asset,horizon,window,name]=m
                    for loss,col in (("MSE","mse_vec"),("QLIKE","qlike_vec")):
                        loss_store[name][loss].append(m[col])
                for loss,col in (("MSE","mse_vec"),("QLIKE","qlike_vec")):
                    statistic,p=hln(ms["NED"][col],ms["TS-only"][col],h)
                    dms[asset,horizon,window,loss]=p
                    rows.append(dict(asset=asset,horizon=horizon,window=window,loss=loss,statistic=statistic,p=p))
            for name in ms:
                means[asset,horizon,name]={k:np.mean([records[asset,horizon,w,name][k] for w in range(1,7)]) for k in ("R2","DA","MSE","QLIKE")}
            for loss in ("MSE","QLIKE"):
                statistic,p=hln(np.concatenate(loss_store["NED"][loss]),np.concatenate(loss_store["TS-only"][loss]),h)
                dms[asset,horizon,"mean",loss]=p
                rows.append(dict(asset=asset,horizon=horizon,window="mean",loss=loss,statistic=statistic,p=p))
    pd.DataFrame(rows).to_csv(ANALYSIS/"dm_hln.csv",index=False)
    for horizon in ("1h","4h"):
        tab=[]
        for asset in ("BTC","ETH"):
            ts=means[asset,horizon,"TS-only"]
            if tab:tab.append(r"\midrule")
            for j,(_,name) in enumerate(MODELS):
                m=means[asset,horizon,name]
                row=[rf"\multirow{{5}}{{*}}{{\textbf{{{asset}}}}}" if j==0 else "",name,f"{m['R2']:.3f}",f"{m['DA']:.3f}"]
                for loss in ("QLIKE","MSE"):
                    cell=f"{m[loss]:.3f}"
                    if name=="NED":cell+=stars(dms[asset,horizon,"mean",loss])+rf"\,({percentage(100*(m[loss]/ts[loss]-1))}\%)"
                    row.append(cell)
                tab.append(row)
        table(f"original_main_{horizon}.tex","llrrrr",r"Asset & Model & $R^2$ & DA & QLIKE & MSE",tab)
    for asset in ("BTC","ETH"):
        for horizon in ("1h","4h"):
            out=[r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",r"Window & Model & $R^2$ & DA & QLIKE & MSE \\",r"\midrule"]
            for w in range(1,7):
                period=f"{2023+(w-1)//2} "+("Jan--Jun" if w%2 else "Jul--Dec")
                for j,(_,name) in enumerate(MODELS):
                    m=records[asset,horizon,w,name];ts=records[asset,horizon,w,"TS-only"]
                    row=[rf"\multirow{{5}}{{*}}{{{period}}}" if j==0 else "",name,f"{m['R2']:.3f}",f"{m['DA']:.3f}"]
                    for loss in ("QLIKE","MSE"):
                        cell=f"{m[loss]:.3f}"
                        if name=="NED":cell+=stars(dms[asset,horizon,w,loss])+rf"\,({percentage(100*(m[loss]/ts[loss]-1))}\%)"
                        row.append(cell)
                    out.append(" & ".join(row)+r" \\")
                if w<6:out.append(r"\midrule")
            out.extend([r"\bottomrule",r"\end{tabular}"])
            (GEN/f"original_appendix_{asset.lower()}_{horizon}.tex").write_text("\n".join(out)+"\n")
    return means


def descriptive():
    pars=pd.read_csv(GEN/"article_parameters.csv")
    pars=pars[(pars.asset=="BTC")&(pars.horizon=="4h")].copy()
    bars=pd.read_pickle("/tmp/crypto-paper-btc-bars.pkl")
    fwd=forward_4h_log_return(bars)
    chunks=[]
    for w in range(1,7):
        p=np.load(ROOT/f"results/main/btc_cryptobert/predictions/window_{w}_4h_predictions.npz")
        chunks.append(pd.DataFrame(dict(actual=p["actual"],ts=p["ts_mlp"].astype(np.float64),ned=p["main_model"].astype(np.float64),
            z=fwd[p["test_idx"]]/np.sqrt(np.maximum(p["ts_mlp"].astype(np.float64),1e-15)))))
    frame=pd.concat(chunks,ignore_index=True)
    assert np.isfinite(frame).all().all()
    frame["decile"]=pd.qcut(frame.z,10,labels=False)+1
    rows=[]
    for d,g in frame.groupby("decile"):
        ts=compute_metrics(g.ts.to_numpy(),g.actual.to_numpy());ned=compute_metrics(g.ned.to_numpy(),g.actual.to_numpy())
        rows.append(dict(decile=d,N=len(g),z_mean=g.z.mean(),R2_TS=ts["R2"],R2_NED=ned["R2"],dR2=ned["R2"]-ts["R2"],
           QLIKE_pct=100*(ned["QLIKE"]/ts["QLIKE"]-1),MSE_pct=100*(ned["MSE"]/ts["MSE"]-1)))
    dec=pd.DataFrame(rows);dec.to_csv(ANALYSIS/"return_zscore_deciles.csv",index=False)
    table("original_deciles.tex","rrrrrrr",r"Decile & $\bar z$ & TS-only $R^2$ & NED $R^2$ & $\Delta R^2$ & QLIKE (\%) & MSE (\%)",
        [[int(r.decile),f"{r.z_mean:.2f}",f"{r.R2_TS:.3f}",f"{r.R2_NED:.3f}",f"{r.dR2:.3f}",f"{r.QLIKE_pct:+.1f}",f"{r.MSE_pct:+.1f}"] for r in dec.itertuples()])
    news=pd.read_parquet(ROOT/"data/news/processed_news.parquet");news=news[news.asset=="BTC"].reset_index(drop=True)
    sent=np.load(ROOT/"data/processed/btc_news_sentiment.npz")
    assert np.array_equal(sent["timestamps"],news.unix_timestamp.to_numpy())
    probs=sent["sentiment_probs"][pars.article_row.to_numpy()]
    label=np.array(["Positive","Negative","Neutral"])[probs.argmax(1)]
    summaries=[];tests=[]
    for threshold in (.5,.6,.7):
        p=pars.copy();p["sentiment"]=np.where(probs.max(1)>threshold,label,"Ambiguous")
        for name,g in p.groupby("sentiment"):
            summaries.append(dict(threshold=threshold,group=name,N=len(g),s_mean=g.s.mean(),s_median=g.s.median(),s_std=g.s.std(),half_median=g.half_life_min.median()))
        for first,second in (("Negative","Positive"),("Ambiguous","Decisive")):
            a=p[p.sentiment==first].s
            b=p[p.sentiment!="Ambiguous"].s if second=="Decisive" else p[p.sentiment==second].s
            stat,pv=mannwhitneyu(a,b,alternative="two-sided")
            tests.append(dict(threshold=threshold,first=first,second=second,U=stat,p=pv))
        if threshold==.6:p.to_csv(ANALYSIS/"sentiment_article_parameters.csv",index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(ANALYSIS/"sentiment_groups.csv",index=False);pd.DataFrame(tests).to_csv(ANALYSIS/"sentiment_tests.csv",index=False)
    tab=[]
    for group in ("Positive","Neutral","Negative","Ambiguous"):
        r=summary[(summary.threshold==.6)&(summary.group==group)].iloc[0]
        tab.append([group+"-decisive" if group!="Ambiguous" else group,int(r.N),f"{r.s_mean:.3f}",f"{r.s_median:.3f}",f"{r.s_std:.3f}",f"{r.half_median:.1f}"])
    table("original_sentiment.tex","lrrrrr",r"Group & $N$ & Mean $s$ & Median $s$ & Std.\ $s$ & Median $h_{1/2}$ (min)",tab)
    # Joint distribution with top and right marginal histograms, as in paper_old.
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig=plt.figure(figsize=(8,6));gs=fig.add_gridspec(2,2,width_ratios=[5,1.1],height_ratios=[1.1,5],hspace=.06,wspace=.06)
    ax=fig.add_subplot(gs[1,0]);top=fig.add_subplot(gs[0,0],sharex=ax);right=fig.add_subplot(gs[1,1],sharey=ax)
    ax.scatter(pars.s,pars.lambda_per_min,s=5,alpha=.22,color="#1f667f",rasterized=True)
    ax.set(xlabel=r"Learned impact magnitude $s$",ylabel=r"Decay rate $\lambda$ (per minute)",yscale="log")
    top.hist(pars.s,bins=45,color="#1f667f",alpha=.7);top.tick_params(labelbottom=False);top.set_ylabel("Count")
    right.hist(pars.lambda_per_min,bins=np.geomspace(.001,.5,45),orientation="horizontal",color="#1f667f",alpha=.7)
    right.tick_params(labelleft=False);right.set_xlabel("Count")
    fig.savefig(FIG/"distribution.pdf",bbox_inches="tight");plt.close(fig)
    rho,pv=spearmanr(pars.s,pars.lambda_per_min)
    stats=dict(N=len(pars),s_mean=pars.s.mean(),s_median=pars.s.median(),half_median=pars.half_life_min.median(),
        half_q25=pars.half_life_min.quantile(.25),half_q75=pars.half_life_min.quantile(.75),half_above_4h_pct=100*(pars.half_life_min>240).mean(),s_lambda_spearman=rho)
    (ANALYSIS/"distribution_summary.json").write_text(json.dumps(stats,indent=2)+"\n")
    case_study(pars,news,bars)
    print(summary[summary.threshold==.6].to_string(index=False))
    print(dec.to_string(index=False))


def case_study(pars,news,bars):
    anchor=pd.Timestamp("2024-03-05 18:30:00",tz="UTC")
    p=np.load(ROOT/"results/main/btc_cryptobert/predictions/window_3_4h_predictions.npz",allow_pickle=True)
    dates=pd.to_datetime(p["datetime"],utc=True);seconds=bars.timestamp.iloc[p["test_idx"]].to_numpy()
    show=(dates>=anchor-pd.Timedelta(hours=12))&(dates<=anchor+pd.Timedelta(hours=12))
    articles=pars[pars.window==3].copy();articles["title"]=news.title.iloc[articles.article_row].to_numpy()
    articles["at_anchor"]=articles.s*np.exp(-articles.lambda_per_min*np.maximum((anchor.timestamp()-articles.timestamp)/60,0))
    eligible=(articles.timestamp>=anchor.timestamp()-43200)&(articles.timestamp<anchor.timestamp())
    selected=articles[eligible].nlargest(3,"at_anchor").copy()
    selected.to_csv(ANALYSIS/"case_study_articles.csv",index=False)
    fig,(ax,low)=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={"height_ratios":[1,1.15]},constrained_layout=True)
    factor=365*24/4
    for key,label,col,style in (("actual","Realized","0.45","-"),("ts_mlp","TS-only","#596573","--"),("main_model","NED","#a02b32","-")):
        ax.plot(dates[show],100*np.sqrt(p[key][show]*factor),label=label,color=col,ls=style,lw=1.6)
    ax.set_ylabel("Annualized volatility (%)");ax.legend(frameon=False,ncol=3,loc="upper left")
    clock=np.arange(int((anchor-pd.Timedelta(hours=12)).timestamp()),int((anchor+pd.Timedelta(hours=12)).timestamp())+60,60)
    clockdt=pd.to_datetime(clock,unit="s",utc=True)
    palette=["#b33435","#dd7666","#eab6a0"]
    cumulative=np.zeros(len(clock));labels=[]
    for i,r in enumerate(selected.itertuples()):
        age=(clock-r.timestamp)/60
        signal=r.s*np.exp(-r.lambda_per_min*np.maximum(age,0))*((age>=0)&(age<=720))
        low.fill_between(clockdt,cumulative,cumulative+signal,color=palette[i],alpha=.8)
        cumulative+=signal
        publication=pd.to_datetime(r.timestamp,unit="s",utc=True)
        ax.axvline(publication,color=palette[i],ls=":",lw=1)
        ax.text(publication,.8*ax.get_ylim()[1],str(i+1),color=palette[i],ha="center",bbox=dict(facecolor="white",edgecolor=palette[i],boxstyle="circle,pad=.15"))
        labels.append(f"{i+1}. Half-life {r.half_life_min:.0f} min; s={r.s:.2f}")
    # Sum of every eligible article, not an allocation of the residual bias.
    all_signal=np.zeros(len(clock))
    for r in articles[(articles.timestamp>=clock[0]-43200)&(articles.timestamp<=clock[-1])].itertuples():
        age=(clock-r.timestamp)/60
        all_signal+=r.s*np.exp(-r.lambda_per_min*np.maximum(age,0))*((age>=0)&(age<=720))
    low.plot(clockdt,all_signal,color="black",lw=1.2,label="All retained articles")
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=palette[i],label=labels[i]) for i in range(3)]+[low.lines[-1]]
    low.legend(handles=handles,loc="upper left",fontsize=8,frameon=True)
    low.set(ylabel=r"News signal $\sum s_k e^{-\lambda_k\Delta t_k}$",xlabel="March 5--6, 2024 (UTC)")
    low.xaxis.set_major_locator(mdates.HourLocator(interval=4));low.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    for a in (ax,low):a.axvline(anchor,color="0.4",ls="--",lw=.7);a.grid(alpha=.15)
    fig.savefig(FIG/"casestudy.pdf",bbox_inches="tight");plt.close(fig)
    table("case_articles.tex","clrr",r"No. & Article & $s$ & $h_{1/2}$ (min)",
        [[i+1,tex(r.title),f"{r.s:.2f}",f"{r.half_life_min:.1f}"] for i,r in enumerate(selected.itertuples())])
    # Make long headlines wrap in a compact supplementary table.
    path=GEN/"case_articles.tex";path.write_text(path.read_text().replace("{clrr}",r"{cp{9cm}rr}"))


def categories():
    config=ANALYSIS/"category_config.json"
    if not config.exists():return False
    labels=pd.read_csv(ANALYSIS/"article_categories.csv")
    pars=pd.read_csv(GEN/"article_parameters.csv")
    pars=pars[(pars.asset=="BTC")&(pars.horizon=="4h")]
    merged=pars.merge(labels,on="article_row",how="left",validate="many_to_one",suffixes=("","_label"))
    assert merged.category.notna().all() and np.array_equal(merged.timestamp,merged.timestamp_label)
    merged.to_csv(ANALYSIS/"category_article_parameters.csv",index=False)
    stats=merged.groupby("category").agg(N=("s","size"),s_mean=("s","mean"),s_std=("s","std"),half_median=("half_life_min","median"),effective_mean=("effective_initial_log_loading","mean")).reset_index()
    stats.to_csv(ANALYSIS/"category_summary.csv",index=False)
    order=stats.sort_values("s_mean",ascending=False)
    display_names={"Regulatory_Legal":"Regulatory/Legal", "Security_Crime":"Security/Crime",
        "Protocol_Infrastructure":"Protocol/Infrastructure", "Opinion_Editorial":"Opinion/Editorial",
        "OnChain_Flows":"On-chain flows", "Newsletter_Series":"Newsletter series"}
    table("original_categories.tex","lrrrrr",r"Category & $N$ & Mean $s$ & Std.\ $s$ & Median $h_{1/2}$ (min) & Mean $as$",
        [[tex(display_names.get(r.category,r.category.replace("_"," "))),int(r.N),f"{r.s_mean:.2f}",f"{r.s_std:.2f}",f"{r.half_median:.1f}",f"{r.effective_mean:.3f}"] for r in order.itertuples()])
    print(order.to_string(index=False));return True


def ablation_tables():
    reports=[json.loads(p.read_text()) for p in ABLATIONS.glob("window_*/*/metrics.json")]
    if len(reports)<72:return False
    df=pd.DataFrame(reports)
    assert len(df)==72 and df.groupby("variant").window.nunique().eq(6).all()
    for r in df.itertuples():
        data=np.load(ABLATIONS/f"window_{r.window}/{r.variant}/predictions.npz")
        metrics=compute_metrics(data["prediction"],data["actual"])
        for key in ("R2","DA","MSE","QLIKE"):assert abs(metrics[key]-getattr(r,key))<1e-8
    mean=df.groupby("variant").agg(R2=("R2","mean"),R2_std=("R2","std"),DA=("DA","mean"),MSE=("MSE","mean"),QLIKE=("QLIKE","mean")).reset_index()
    df.to_csv(ABLATIONS/"metrics_per_window.csv",index=False);mean.to_csv(ABLATIONS/"metrics_mean.csv",index=False)
    old=pd.read_csv(GEN/"metrics_mean.csv");ts=old[(old.asset=="BTC")&(old.horizon=="4h")&(old.model=="TS MLP")].iloc[0]
    by=mean.set_index("variant")
    def metric(loss,value):return f"{value:.3f}"+rf"\,({percentage(100*(value/ts[loss]-1))}\%)"
    rows=[["TS-only (no news)",f"{ts.R2:.3f}",f"{ts.MSE:.3f}",f"{ts.QLIKE:.3f}",f"{ts.DA:.3f}"]]
    for key,label in (("constant","Constant (no decay)"),("power","Power-law"),("exponential","Exponential"),("linear","Linear")):
        r=by.loc[key];rows.append([label,rf"{r.R2:.3f}\,$\pm${r.R2_std:.3f}",metric("MSE",r.MSE),metric("QLIKE",r.QLIKE),f"{r.DA:.3f}"])
    table("original_decay.tex","lrrrr",r"Decay function & $R^2$ & MSE & QLIKE & DA",rows)
    rows=[["TS-only (no news)",f"{ts.R2:.3f}",f"{ts.MSE:.3f}",f"{ts.QLIKE:.3f}"]]
    for key,label in (("news_count","+ News count"),("sentiment_recency",r"+ Sentiment $\times$ recency"),("daily_sentiment",r"+ Daily agg.\ sentiment"),("embedding_mean","+ CryptoBERT 768-dim mean-pool"),("sentiment_decay","+ Sentiment, learned decay"),("pca256_decay","+ CryptoBERT 256-dim (PCA)"),("exponential","+ CryptoBERT 768-dim")):
        r=by.loc[key];rows.append([label,f"{r.R2:.3f}",metric("MSE",r.MSE),metric("QLIKE",r.QLIKE)])
    table("original_representations.tex","lrrr",r"Representation & $R^2$ & MSE & QLIKE",rows)
    path=GEN/"original_representations.tex";s=path.read_text().replace("TS-only (no news)",r"\multicolumn{4}{@{}l}{\textit{A. Without learned decay}} \\[3pt]"+"\nTS-only (no news)",1)
    s=s.replace("+ Sentiment, learned decay",r"\addlinespace\multicolumn{4}{@{}l}{\textit{B. With learned $(s,\lambda)$ decay}} \\[3pt]"+"\n+ Sentiment")
    path.write_text(s)
    table("original_lookback.tex","lrrrr",r"Lookback & $R^2$ & DA & QLIKE & MSE",
        [[label,f"{by.loc[key,'R2']:.3f}",f"{by.loc[key,'DA']:.3f}",f"{by.loc[key,'QLIKE']:.3f}",f"{by.loc[key,'MSE']:.3f}"] for key,label in (("lookback_6h","6 hours"),("exponential","12 hours"),("lookback_24h","24 hours"))])
    print(mean.to_string(index=False));return True


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--require-complete",action="store_true");args=parser.parse_args()
    main_tables();descriptive();c=categories();a=ablation_tables()
    if args.require_complete:assert c and a,"Required training/classification results still incomplete"
    print(json.dumps(dict(main=True,descriptive=True,categories=c,ablations=a)))
