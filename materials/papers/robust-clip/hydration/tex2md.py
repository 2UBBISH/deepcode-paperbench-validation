"""Convert the sections missing from PaperBench's robust-clip paper.md out of the arXiv LaTeX source (2402.12336v2),
in PaperBench's own md dialect (\\section*{N. Title}, inline LaTeX math, author-year citations)."""
import re, sys, os
SRC=sys.argv[1]; tex=open(f"{SRC}/main_icml.tex",encoding="utf-8").read(); bbl=open(f"{SRC}/main_icml.bbl",encoding="utf-8").read()
# --- citations: bibitem[Author et al.(2022)…]{key} → "Author et al., 2022"
cites={}
for m in re.finditer(r"\\bibitem\[\{?(.*?)\}?\]\{(.*?)\}", bbl, re.S):
    lab=m.group(1); key=m.group(2)
    lab=re.sub(r"\{\\natexlab\{([a-z])\}\}", r"\1", lab); lab=re.sub(r"\{\\c\{c\}\}", "ç", lab); lab=re.sub(r"[{}]", "", lab)
    mm=re.match(r"(.*?)\((\d{4}[a-z]?)\)", lab.replace("~"," ").replace("\n"," "))
    cites[key]=(mm.group(1).strip(), mm.group(2)) if mm else (lab,"")
# --- macros
MACROS={r"\ours":"FARE", r"\tecoa":"TeCoA", r"\clip":"CLIP", r"\llava":"LLaVA", r"\openf":"OpenFlamingo", r"\imnet":"ImageNet",
        r"\oursfour":"FARE$^{4}$", r"\ourstwo":"FARE$^{2}$", r"\tecoafour":"TeCoA$^{4}$", r"\tecoatwo":"TeCoA$^{2}$",
        r"\aatt":"AutoAttack", r"\eg":"e.g.", r"\ie":"i.e.", r"\cf":"cf.", r"\etal":"et al.", r"\phift":"\\phi_{\\text{FT}}", r"\phiorg":"\\phi_{\\text{org}}"}
for m in re.finditer(r"\\newcommand\{(\\[a-zA-Z]+)\}\{(.*?)\}\s*$", tex, re.M):
    k,v=m.group(1),m.group(2).replace(r"\xspace","")
    MACROS.setdefault(k, v)
# labels → numbers (tables/figures as they appear in the ICML pdf; sections; equations in source order)
REF={"tab:robust-vlm":"Table 1","tab:transfer-att":"Table 2","tab:targeted-attack":"Table 3","tab:zero-shot":"Table 4","tab:pope":"Table 5","tab:sqa":"Table 6","tab:jailbreaks":"Table 7",
     "fig:teaser":"Fig. 1","fig:teaser-attack":"Fig. 2","fig:targeted-attack":"Fig. 3","fig:pope":"Fig. 4",
     "sec:unsup-adv-ft":"Sec. 3","sec:tecoa-def":"Sec. 3.2","sec:attack-vlm-untargeted":"Sec. 4.1","sec:attack-stealthy-targeted":"Sec. 4.2","sec:zero-shot":"Sec. 4.3","sec:other":"Sec. 4.4","sec:jailbreak":"Sec. 4.4","sec:sqa":"App. C.2",
     "thm:emb_distance":"Theorem 3.1","app:exp-detail":"App. B","app:attack-detail":"App. B.8","app:zero-shot":"App. B.10","app:proof":"App. A","app:ablation":"App. B.3","app:tecoa-comparison":"App. B.5","app:loss-ablation":"App. B.4","app:untargeted-attack-detail":"App. B.6","app:hallucination":"App. C.1"}
eqs=[m.group(1) for m in re.finditer(r"\\label\{(eq:[^}]+)\}", tex)]
for i,l in enumerate(eqs): REF[l]=f"({i+1})"
def sub_cites(s):
    def one(m, paren):
        keys=[k.strip() for k in m.group(1).split(",")]
        parts=[]
        for k in keys:
            a,y=cites.get(k,(k,"")); parts.append(f"{a}, {y}" if paren else f"{a} ({y})")
        return ("("+"; ".join(parts)+")") if paren else "; ".join(parts)
    s=re.sub(r"\\citet\{([^}]+)\}", lambda m: one(m,False), s)
    s=re.sub(r"\\cite[pt]?\*?\{([^}]+)\}", lambda m: one(m,True), s)
    return s
def conv(s):
    s=re.sub(r"(?<!\\)%.*", "", s)                                 # comments
    s=re.sub(r"\\input\{[^}]*\}|\\afterpage\{[^}]*\}|\\vspace\{[^}]*\}|\\clearpage|\\newpage", "", s)
    s=re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", s, flags=re.S)
    s=re.sub(r"\\label\{[^}]*\}", "", s); s=re.sub(r"\n\}\s*\n", "\n", s)
    s=re.sub(r"\\norm\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\\left\\|\1\\right\\|", s)
    s=re.sub(r"\\inner\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\\left\\langle\1\\right\\rangle", s)
    s=s.replace("\\R^","\\mathbb{R}^").replace("\\Big(","\\left(").replace("\\Big)","\\right)").replace("\\argmax","\\operatorname{argmax}").replace("\\argmin","\\operatorname{argmin}")
    s=re.sub(r"\\textrm\{([^{}]*)\}", r"\\mathrm{\1}", s); s=re.sub(r"\\noindent\s*", "", s); s=re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", s)
    s=re.sub(r"\\Cref\{([^}]+)\}", lambda m: REF.get(m.group(1), m.group(1)), s)
    for k in sorted(MACROS, key=len, reverse=True): s=re.sub(re.escape(k)+r"(?![a-zA-Z])", MACROS[k].replace("\\","\\\\"), s)
    s=sub_cites(s)
    s=re.sub(r"(Eqs?|Eq)\.?~?\((\\ref\{[^}]+\}(?:,\s*\\ref\{[^}]+\})*)\)", lambda m: m.group(1)+". "+", ".join(REF.get(r,r) for r in re.findall(r"\\ref\{([^}]+)\}", m.group(2))), s)
    s=re.sub(r"\\eqref\{([^}]+)\}", lambda m: REF.get(m.group(1), m.group(1)), s)
    s=re.sub(r"(Table|Tables|Fig|Figs|Sec|App|Theorem)\.?~?\\ref\{([^}]+)\}", lambda m: REF.get(m.group(2), m.group(2)), s)
    s=re.sub(r"~?\\ref\{([^}]+)\}", lambda m: REF.get(m.group(1), m.group(1)), s)
    s=re.sub(r"\\paragraph\{([^}]*)\}", r"\n\1", s)
    s=re.sub(r"\\item\[\\textbf\{([^}]*)\}\]", r"\n\1", s); s=re.sub(r"\\item\[([^\]]*)\]", r"\n\1", s); s=s.replace(r"\item","\n-")
    s=re.sub(r"\\begin\{description\}(\[[^\]]*\])?|\\end\{description\}|\\begin\{itemize\}|\\end\{itemize\}", "", s)
    s=re.sub(r"\\begin\{theorem\}", "**Theorem 3.1.** ", s); s=re.sub(r"\\end\{theorem\}", "", s)
    s=re.sub(r"\\begin\{proof\}", "*Proof.* ", s); s=re.sub(r"\\end\{proof\}", "", s)
    s=re.sub(r"\\(?:textit|emph|textbf|textrm|text|mbox)\{([^{}]*)\}", r"\1", s)
    s=re.sub(r"\\(?:textit|emph|textbf)\{([^{}]*)\}", r"\1", s)
    s=re.sub(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", lambda m: "\n$$\n"+m.group(1).strip().replace("&","")+"\n$$\n", s, flags=re.S)
    s=re.sub(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", lambda m: "\n$$\n"+m.group(1).strip()+"\n$$\n", s, flags=re.S)
    s=re.sub(r"\\\[(.*?)\\\]", lambda m: "\n$$\n"+m.group(1).strip()+"\n$$\n", s, flags=re.S)
    s=s.replace(r"\nicefrac","\\frac").replace("~"," ").replace(r"\&","&").replace(r"\%","%").replace(r"\,"," ").replace(r"\\ "," ")
    s=re.sub(r"\\footnote\{([^{}]*)\}", r" (\1)", s)
    s=re.sub(r"[ \t]+\n", "\n", s); s=re.sub(r"\n{3,}", "\n\n", s); s=re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()
def section(start_pat, end_pat):
    i=re.search(start_pat, tex).start(); j=re.search(end_pat, tex[i+10:]).start()+i+10
    return tex[i:j]
def heading(line):  # \section{Related Work} → \section*{2. Related Work}
    return line
out=[]
# §1 remainder: from the sentence the md cut in half to the end of the introduction
s1=section(r"Given the flexibility and effectiveness", r"\\section\{Related Work\}")
out.append(conv(s1))
# §2, §3 (+3.1–3.3), §4 intro
body=section(r"\\section\{Related Work\}", r"\\subsection\{Quantitative Robustness Evaluation of LVLMs\}")
body=body.replace(r"\section{Related Work}", r"\section*{2. Related Work}")
body=body.replace(r"\section{Unsupervised Adversarial Fine-Tuning for CLIP}", r"\section*{3. Unsupervised Adversarial Fine-Tuning for CLIP}")
body=body.replace(r"\subsection{Robustness of  CLIP as Zero-Shot Classifier}", r"\subsection*{3.1. Robustness of CLIP as Zero-Shot Classifier}")
body=body.replace(r"\subsection{Supervised Adversarial Fine-Tuning}", r"\subsection*{3.2. Supervised Adversarial Fine-Tuning}")
body=body.replace(r"\subsection{Unsupervised Adversarial Fine-Tuning of the Image Embedding}", r"\subsection*{3.3. Unsupervised Adversarial Fine-Tuning of the Image Embedding}")
body=body.replace(r"\section{Experiments}", r"\section*{4. Experiments}")
out.append(conv(body))
s42=section(r"\\subsection\{Stealthy Targeted Attacks on LVLMs\}", r"\\subsection\{Evaluation of Zero-Shot Classification\}").replace(r"\subsection{Stealthy Targeted Attacks on LVLMs}", r"\subsection*{4.2. Stealthy Targeted Attacks on LVLMs}")
s43=section(r"\\subsection\{Evaluation of Zero-Shot Classification\}", r"\\subsection\{Performance on Other Tasks\}").replace(r"\subsection{Evaluation of Zero-Shot Classification}", r"\subsection*{4.3. Evaluation of Zero-Shot Classification}")
open(f"{SRC}/../part_intro_to_4.md","w").write(out[0]+"\n\n"+out[1]+"\n")
open(f"{SRC}/../part_4_2.md","w").write(conv(s42)+"\n"); open(f"{SRC}/../part_4_3.md","w").write(conv(s43)+"\n")
left=set(re.findall(r"\\[a-zA-Z]+", out[0]+out[1]+conv(s42)+conv(s43)))
print("unconverted commands:", sorted(c for c in left if c not in {"\\section*","\\subsection*","\\frac","\\left","\\right","\\phi","\\psi","\\epsilon","\\ell","\\infty","\\max","\\min","\\sum","\\log","\\leq","\\geq","\\in","\\mathbb","\\mathcal","\\mathrm","\\text","\\argmin","\\rightarrow","\\|","\\alpha","\\times","\\cdot","\\langle","\\rangle","\\top","\\hat","\\tilde","\\ldots","\\cos","\\quad","\\ge","\\le","\\ne","\\neq","\\lim","\\nolimits","\\limits","\\mathop","\\rm","\\pm","\\delta","\\,","\\dots"}))
