"""Build human-readable LaTeX review and verified-answer packets."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

SUBJECTS = ("physics", "chemistry", "biology")
CHOICE_LABELS = "ABCD"
ELEMENTS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}
MATH_SPAN = re.compile(r"(\\+\(.*?\\+\)|\\+\[.*?\\+\]|\$[^$\n]+\$)", re.DOTALL)
TAG = re.compile(r"<(SMILES|IUPAC|INCHI)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
MALFORMED_TAG = re.compile(
    r"<(SMILES|IUPAC|INCHI)>(.*?)(?:</\1>|</\1(?=,))",
    re.DOTALL | re.IGNORECASE,
)
MARKDOWN_TABLE = re.compile(r"(?:^\|.*\|\s*(?:\n|$))+", re.MULTILINE)
MARKDOWN_BULLETS = re.compile(r"(?:^- .*(?:\n|$))+", re.MULTILINE)
ANSWER_INSTRUCTION = re.compile(
    r"\s*Think step by step and solve the problem below\..*?"
    r"without any extra commentary or providing multiple answer attempts\.\s*$",
    re.DOTALL,
)

MATH_UNICODE = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\varepsilon",
    "ϵ": r"\epsilon",
    "η": r"\eta",
    "θ": r"\theta",
    "µ": r"\mu",
    "μ": r"\mu",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "χ": r"\chi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Ω": r"\Omega",
    "∂": r"\partial",
    "×": r"\times",
    "−": "-",
    "≈": r"\approx",
    "≠": r"\ne",
    "≪": r"\ll",
    "≫": r"\gg",
    "·": r"\cdot",
    "〈": r"\langle",
    "〉": r"\rangle",
    "𝑓": "f",
    "𝑃": "P",
    "°": r"^{\circ}",
    "¹": "^{1}",
    "²": "^{2}",
    "³": "^{3}",
    "⁻": "^{-}",
    "₀": "_{0}",
    "₁": "_{1}",
    "₂": "_{2}",
    "₃": "_{3}",
    "₄": "_{4}",
    "₅": "_{5}",
    "₆": "_{6}",
    "₇": "_{7}",
    "₈": "_{8}",
    "₉": "_{9}",
    "\u00a0": " ",
}

TEXT_UNICODE = {
    "“": "``",
    "”": "''",
    "’": "'",
    "–": "--",
    "—": "---",
    "，": ",",
    "\u00a0": " ",
    **{
        source: rf"\({replacement}\)"
        for source, replacement in MATH_UNICODE.items()
        if source != "\u00a0"
    },
}

PROSE_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}

BARE_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "eta": "η",
    "theta": "θ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "chi": "χ",
    "omega": "ω",
}

# The review notes contain equations written as plain-text prose. Curating
# their presentation here keeps the wording intact while grouping each
# complete expression into one math span.
REVIEW_GOLD_LATEX: dict[str, str] = {
    "f10254b9-f0a1-407f-8af9-3169e23eee5e": (
        r"Interpret \(\omega>0\) as the angular-speed magnitude, so "
        r"\(\vec\omega=-\omega\hat y\) fixes the rotation direction. Immediately "
        r"after release, \(\vec v\simeq-gt\,\hat z\), and therefore "
        r"\(S(\vec v\times\vec\omega)\simeq-Sg\omega t\,\hat x\); the cylinder "
        r"must initially move toward negative \(x\). With \(\alpha=k\omega\): "
        r"\(\ddot{x}=\alpha\dot{z}\) and \(\ddot{z}=-\alpha\dot{x}-g\). "
        r"Thus \(x(t)=a(\sin\alpha t-\alpha t)\), "
        r"\(z(t)=a(\cos\alpha t-1)\), and \(a=g/(k^2\omega^2)\). "
        r"During the first descent after release, \(0\leq\alpha t\leq\pi\), "
        r"eliminating \(t\) gives "
        r"\(x(z)=\sqrt{-z(2a+z)}-a\arccos(1+z/a)\). "
        r"This result is negative, consistent with the short-time expansion "
        r"\(x\simeq-gk\omega t^3/6\). The reference has the opposite sign. If "
        r"\(\omega<0\), the reference has the correct sign, but that interpretation "
        r"conflicts with the standard convention that \(\omega\) is a magnitude."
    ),
    "e1b0b588-915d-46ad-990e-ba5aa554053e": (
        r"For an axisymmetric purely azimuthal velocity field "
        r"\(\mathbf{u}=v(r,t)\mathbf{e}_{\theta}\), with "
        r"\(v=(\Gamma/2\pi)g\), the \(\theta\) component of Navier--Stokes is "
        r"\(v_t=(\eta/\rho)(v_{rr}+v_r/r-v/r^2)\). Thus "
        r"\(g_t=(\eta/\rho)(g_{rr}+g_r/r-g/r^2)\). The reference answer "
        r"instead matches the diffusion equation for approximately \(rg\), not for "
        r"\(g\) as defined."
    ),
    "1dba0086-bca8-4b25-8843-e6e428f9284d": (
        r"The reference answer is correct if the Type B scale error is independent "
        r"for each of the \(N\) readings, giving variance contribution "
        r"\((a^2/12)/N\) to the mean. However, the stem uses a singular \(X_B\) "
        r"added to all results, which naturally reads as a common offset; then the "
        r"uncertainty would be \(\sqrt{s^2/N+a^2/12}\)."
    ),
    "84bba88d-2aee-4fcb-baab-4c6eea17b0d8": (
        r"The reference has dimensionful logarithm arguments and an erroneous power "
        r"of \(c_1+c_2\). With \(\tau=0\) and constant reservoir heat capacities, "
        r"energy conservation gives "
        r"\(T_f=(c_1T_1+c_2T_2)/(c_1+c_2)\), so the entropy change is "
        r"\(c_1\ln(T_f/T_1)+c_2\ln(T_f/T_2)\)."
    ),
    "bdb3fc5f-9374-4e37-9af9-6036e3e29093": (
        r"Let \(\beta=v/c\). In the conductor rest frame, "
        r"\(E'_0=\gamma(1-\beta)E_0\) and "
        r"\(\omega'=\gamma(1-\beta)\omega\); a stationary perfect conductor has "
        r"\(K'_y=2E'_0/(\mu_0c)\cos(\omega't')\). On the lab surface "
        r"\(x=vt\), \(t'=t/\gamma\), and the material sheet current transforms as "
        r"\(K_y=K'_y/\gamma\), giving "
        r"\(2(1-\beta)E_0/(\mu_0c)\cos[\omega(1-\beta)t]\,\hat{\mathbf y}\). "
        r"Even the magnetic-jump convention gives "
        r"\(2E_0/[\mu_0c(1+\beta)]\cos[\omega(1-\beta)t]\,\hat{\mathbf y}\). "
        r"The reference phase and amplitude are therefore not correct."
    ),
    "dcc38501-c38e-4a8e-a069-1a401d6432c0": (
        r"The reference answer is correct only for the likely intended variant where "
        r"the final speed \(4c/5\) is measured in the initial pre-stage-1 frame. "
        r"As written, \(\mathcal F\) is comoving after stage 1, so stage 2 alone "
        r"gives rapidity \(\ln 3\), and the total mass ratio \(4\) gives stage-1 "
        r"rapidity \(\ln(4/3)\). The angle \(\alpha\) is then arbitrary; there is "
        r"no positive minimum, only infimum \(0\)."
    ),
    "2c46387b-36e6-43cb-9e73-dbf1341bfafd": (
        r"The value \(1.56\) follows if \(O\) is the fixed center of mass and "
        r"\(P_2\) and \(P_3\) are mirror-symmetric about \(OP\): "
        r"\(OP=(2/3)PP_2\cos(11.6^\circ)\), which gives "
        r"\(OP_2\approx92.8L\). Those standard figure-eight choreography "
        r"assumptions are not stated explicitly, so the literal geometric data do "
        r"not uniquely determine \(OP_2\)."
    ),
    "c982b5bc-0fb8-4f12-94fa-f05715209ead": (
        r"Intended shortcut: \(\ce{Ca(OH)2}\) absorbs \(50\,\mathrm{cm^3}\) "
        r"\(\ce{CO2}\); \(500-440=60\,\mathrm{cm^3}\) is treated as "
        r"\(\ce{H2O}\). For \(\mathrm{C}_n\mathrm{H}_{2n+2}\mathrm{O}_2\), products are "
        r"\(n\,\ce{CO2}\) and \((n+1)\,\ce{H2O}\), so "
        r"\((n+1)/n=60/50\) gives \(n=5\), i.e. \(\ce{C5H12O2}\). "
        r"This does not satisfy all stated volume data."
    ),
    "85b4f862-d881-4a79-8c5d-3e927b486b71": (
        r"The titration gives \(0.003750\,\mathrm{mol}\) total \(\ce{Fe}\) atoms. "
        r"Solving \(x+y=0.225\,\mathrm{g}\) and "
        r"\(x/55.845+2y/159.687=0.003750\) gives about "
        r"\(0.17316\,\mathrm{g}\) \(\ce{Fe}\) and "
        r"\(0.05184\,\mathrm{g}\) \(\ce{Fe2O3}\), or \(76.96\%\) "
        r"\(\ce{Fe}\) and \(23.04\%\) \(\ce{Fe2O3}\). The supplied "
        r"\(76.89\%/23.11\%\) does not follow from standard molar masses."
    ),
    "b7337452-ab08-43e5-9f3b-419107415280": (
        r"Ionization jumps after \(I_2\) for \(D\) and after \(I_4\) for \(E\) "
        r"imply \(D\) is group 2 and \(E\) is group 14, so \(L=DO\) and "
        r"\(M=EO_2\). \(D_2EO_4\) is a plausible formal "
        r"\(2DO+EO_2\) product, but it is not uniquely determined."
    ),
    "cc251e69-6a32-4ed7-9315-beee80404866": (
        r"The reference aniline is the expected product for the standard Curtius "
        r"sequence using ethyl chloroformate/\(\ce{Et3N}\)/\(\ce{NaN3}\) followed "
        r"by hydrolysis. However, the stated reagent is consistently ethyl "
        r"2-chloroacetate, not ethyl chloroformate. That reagent would not form the "
        r"mixed anhydride/acyl azide needed for the Curtius rearrangement; "
        r"carboxylate alkylation or recovery/hydrolysis pathways are more plausible. "
        r"Thus the gold is not valid under the literal stem."
    ),
    "f44b0903-1c29-44a5-9f9b-b6ccdd164e1c": (
        r"The reference appears to intend docosahexaenoic acid "
        r"(DHA, \(\mathrm{C}_{22}{:}6\)), but the written name "
        r"\emph{Decosahexanoic acid} is nonstandard. Standard "
        r"monoglucosyl-phosphatidylglycerol mass accounting gives DHA parent mass "
        r"about \(1028.6\) exact / \(1029.3\) average, not \(1026.5\)."
    ),
    "3d762ba1-3958-453b-b34e-fe24c2822994": (
        r"If \(K_s\) is taken as the \(ES\) dissociation constant and \(J_s\) as "
        r"the association constant for \(ES+S\rightarrow SES\), then "
        r"\([ES]=[E][S]/K_s\), \([SES]=J_s[ES][S]\), and "
        r"\(v=k[E]_0[S]/(K_s+[S]+J_s[S]^2)\), so the intended \(K_M\) term is "
        r"\(K_s+J_s[S]^2\). However, the reference answer is malformed and the "
        r"stem does not define the equilibrium-constant conventions."
    ),
}

REVIEW_STEM_LATEX: dict[str, str] = {
    "f10254b9-f0a1-407f-8af9-3169e23eee5e": (
        r"The dynamics are clear, but \(x(z)\) is multi-valued over the full "
        r"cycloidal motion unless a time/branch interval is specified."
    ),
    "84bba88d-2aee-4fcb-baab-4c6eea17b0d8": (
        r"The intended thermodynamic endpoint is unique: no external work, isolated "
        r"pair of reservoirs, final common temperature set by energy conservation. "
        r"This assumes \(c_1\) and \(c_2\) are constant heat capacities, as implied "
        r"by the requested variables."
    ),
    "bdb3fc5f-9374-4e37-9af9-6036e3e29093": (
        r"The physical setup is mostly clear, but for a moving boundary the stem "
        r"should specify material/free surface current versus the effective "
        r"\(\mathbf n\times\mathbf H\) magnetic-jump sheet current. Under the "
        r"natural material-current reading it has a unique answer, but not the "
        r"referenced one."
    ),
    "dcc38501-c38e-4a8e-a069-1a401d6432c0": (
        r"The stated frame for the final velocity removes the intended "
        r"Lorentz-composition constraint on \(\alpha\), so the question does not "
        r"determine the reference answer or any unique minimum in \((0,\pi]\)."
    ),
    "c982b5bc-0fb8-4f12-94fa-f05715209ead": (
        r"The initial \(500\,\mathrm{cm^3}\) and final \(500\,\mathrm{cm^3}\) at "
        r"\(300^\circ\mathrm C\) are inconsistent for any nonzero saturated acyclic "
        r"diol. Combustion of \(\mathrm{C}_n\mathrm{H}_{2n+2}\mathrm{O}_2\) "
        r"changes hot gas volume by "
        r"\((n+1)/2\) fuel-volumes. For the intended \(n=5\), the product data "
        r"imply \(10\,\mathrm{cm^3}\) fuel and initial total "
        r"\(470\,\mathrm{cm^3}\), not \(500\,\mathrm{cm^3}\). The "
        r"\(300^\circ\mathrm C\)-to-room-temperature volume comparison is also "
        r"physically ill-posed."
    ),
    "b7337452-ab08-43e5-9f3b-419107415280": (
        r"The stated \(D{:}E>1\) condition excludes the ordinary \(1{:}1\) product "
        r"\(DEO_3\), but still leaves many neutral oxide-adduct formulas "
        r"\(D_aE_bO_{a+2b}\) with \(a/b>1\), e.g. \(D_3EO_5\) or \(D_3E_2O_7\)."
    ),
    "f44b0903-1c29-44a5-9f9b-b6ccdd164e1c": (
        r"Assuming glucosidase removes one hexose residue, parent mass is "
        r"\(PG+\ce{C6H10O5}\). For identical \(\mathrm C_{n}{:}u\) acyl chains, "
        r"average mass is \(28.054n-4.032u+436.259\). The stated \(1026.5\) gives "
        r"no integer \(\mathrm C_{n}{:}u\) solution and does not uniquely imply DHA."
    ),
    "3d762ba1-3958-453b-b34e-fe24c2822994": (
        r"The phrase \emph{equilibrium constant} is not enough to fix whether \(K_s\) "
        r"and \(J_s\) are association or dissociation constants. Standard chemistry "
        r"convention for the reactions as written would give a different expression, "
        r"e.g. \(K_M=K_s^{-1}+J_s[S]^2\). Thus the intended answer is not unique."
    ),
}

REVIEW_REFERENCE_LATEX: dict[str, str] = {
    "85b4f862-d881-4a79-8c5d-3e927b486b71": (
        r"\(76.89\%\) elemental iron (\(\ce{Fe}\)) and \(23.11\%\) iron(III) "
        r"oxide (\(\ce{Fe2O3}\))."
    ),
    "3d762ba1-3958-453b-b34e-fe24c2822994": (r"\(K_M=K_s^{-1}+J_s[S]^2\)"),
}

INCORRECT_REVIEW_IDS = frozenset(
    {
        "e1b0b588-915d-46ad-990e-ba5aa554053e",
        "710c1336-0995-4d45-bfea-48ff475f255f",
        "f9fdb988-7a10-4510-a8ab-d011e663be3f",
        "84bba88d-2aee-4fcb-baab-4c6eea17b0d8",
        "3a998a27-4c4b-47c6-bd2c-dd1c9deb3d63",
        "dcc38501-c38e-4a8e-a069-1a401d6432c0",
        "bdb3fc5f-9374-4e37-9af9-6036e3e29093",
        "85b4f862-d881-4a79-8c5d-3e927b486b71",
        "cc251e69-6a32-4ed7-9315-beee80404866",
        "fcbd76ad-1b45-4f40-ad9d-d2bc5016f02c",
    }
)

REVIEW_ANALYSIS_LATEX: dict[str, str] = {
    "f10254b9-f0a1-407f-8af9-3169e23eee5e": (
        r"Interpret \(\omega>0\) as the angular-speed magnitude, so the stated "
        r"vector \(\vec\omega=-\omega\hat y\) fixes the rotation direction. "
        r"Immediately after release, \(\vec v\simeq-gt\,\hat z\), and hence"
        r"\[S(\vec v\times\vec\omega)\simeq-Sg\omega t\,\hat x.\]"
        r"The cylinder must therefore initially move toward negative \(x\). Let "
        r"\(\alpha=k\omega>0\) and \(a=g/(k^2\omega^2)\). The equations of motion "
        r"are \(\ddot{x}=\alpha\dot{z}\) and \(\ddot{z}=-\alpha\dot{x}-g\), "
        r"with solutions \(x(t)=a(\sin\alpha t-\alpha t)\) and "
        r"\(z(t)=a(\cos\alpha t-1)\). During the first descent, define"
        r"\[q=\arccos(1+z/a),\qquad 0\leq q\leq\pi.\]"
        r"Because \(\sqrt{-z(2a+z)}=a\sin q\), the supplied answer is "
        r"\(x_{\mathrm{supplied}}=a(q-\sin q)\geq0\), whereas the stated dynamics "
        r"give \(x=a(\sin q-q)\leq0\). If \(\omega\) is instead treated as a "
        r"signed scalar, the first-descent result is"
        r"\[x(z)=\operatorname{sgn}(\omega)\left[\sqrt{-z(2a+z)}"
        r"-a\arccos(1+z/a)\right].\]"
        r"For \(\omega<0\), this equals the supplied answer. That interpretation, "
        r"however, conflicts with the standard convention that \(\omega\) in "
        r"\(\vec\omega=-\omega\hat y\) is a positive magnitude. The question does "
        r"not state this convention explicitly. In addition, later portions of the "
        r"trajectory revisit the same value of \(z\), so \(x\) is not globally a "
        r"single-valued function of \(z\)."
    ),
    "e1b0b588-915d-46ad-990e-ba5aa554053e": (
        r"For an axisymmetric, purely azimuthal velocity field "
        r"\(\mathbf{u}=v(r,t)\mathbf{e}_{\theta}\), with \(v=(\Gamma/2\pi)g\), "
        r"the \(\theta\)-component of the Navier--Stokes equation is"
        r"\[v_t=\frac{\eta}{\rho}\left(v_{rr}+\frac{v_r}{r}-\frac{v}{r^2}\right).\]"
        r"Therefore \(g_t=(\eta/\rho)(g_{rr}+g_r/r-g/r^2)\). The supplied answer "
        r"instead resembles an equation for a circulation-related variable, not for "
        r"\(g\) as defined. The definition of \(g\) and the stated symmetries are "
        r"sufficient for a unique reduction."
    ),
    "710c1336-0995-4d45-bfea-48ff475f255f": (
        r"In the sphere's rest frame, \(\vec E'=\vec v\times\vec B=vB_0\hat x\). "
        r"A dielectric sphere in a uniform field has the exterior scattered field"
        r"\[\frac{\varepsilon-\varepsilon_0}{\varepsilon+2\varepsilon_0}vB_0a^3"
        r"\frac{3(\hat r\cdot\hat x)\hat r-\hat x}{r^3}.\]"
        r"Transforming back to the laboratory frame cancels the uniform field. In the "
        r"first vector term, the supplied answer uses \(\hat x\) where the dipole "
        r"field requires \(\hat r\). The result is unique if the surrounding medium "
        r"is vacuum and \(r>a\)."
    ),
    "f9fdb988-7a10-4510-a8ab-d011e663be3f": (
        r"Under the standard interpretation, the sheet moment density is "
        r"\(\sigma(\sin kx\,\hat x+\cos kx\,\hat y)\) on \(y=0\). Solving the "
        r"scalar-potential jump conditions gives \(\vec B=0\) for \(y>0\) and "
        r"\(\vec B=\mu_0\sigma k e^{ky}(-\sin kx\,\hat x+\cos kx\,\hat y)\) for "
        r"\(y<0\), assuming \(k>0\). The factor \(e^{-ky}\) in the supplied answer "
        r"grows as \(y\to-\infty\) and gives "
        r"\(\nabla\cdot\vec B=-2\mu_0\sigma k^2 e^{-ky}\cos(kx)\neq0\) away from "
        r"the sheet. The question leaves the infinite-sheet, sign, and handedness "
        r"conventions implicit, but these conventions do not make the supplied field valid."
    ),
    "84bba88d-2aee-4fcb-baab-4c6eea17b0d8": (
        r"The logarithms in the supplied answer have dimensionful arguments, and the "
        r"power \(c_1+c_2\) is erroneous. With \(\tau=0\) and constant reservoir "
        r"heat capacities, energy conservation gives"
        r"\[T_f=\frac{c_1T_1+c_2T_2}{c_1+c_2},\qquad"
        r"\Delta S=c_1\ln\!\left(\frac{T_f}{T_1}\right)"
        r"+c_2\ln\!\left(\frac{T_f}{T_2}\right).\]"
        r"This result assumes that \(c_1\) and \(c_2\) are constant, as implied by "
        r"the variables requested in the question."
    ),
    "3a998a27-4c4b-47c6-bd2c-dd1c9deb3d63": (
        r"Under the usual rotating-frame convention, with the rotation appearing "
        r"counter-clockwise when viewed from above the local hemisphere, the local "
        r"vertical component is \(\omega\sin\phi\). The Coriolis term "
        r"\(-2\boldsymbol\omega\times\mathbf v\) deflects a radially moving particle "
        r"to the right. Because \(\theta\) is measured clockwise, this gives "
        r"\(+2\omega\sin\phi\,\dot r\), rather than the negative sign in the supplied "
        r"answer. In addition, a particle constrained only to the rotating sphere "
        r"experiences tangential centrifugal terms. The question would need to state a "
        r"local Coriolis-only approximation to exclude them."
    ),
    "dcc38501-c38e-4a8e-a069-1a401d6432c0": (
        r"The supplied answer follows only if the final speed \(4c/5\) is measured in "
        r"the initial, pre-stage-1 frame. As written, \(\mathcal F\) is comoving with "
        r"the rocket after stage 1, so stage 2 alone gives rapidity \(\ln 3\), while "
        r"the total mass ratio of 4 gives a stage-1 rapidity of \(\ln(4/3)\). The "
        r"angle \(\alpha\) is then unconstrained: there is no positive minimum in "
        r"\((0,\pi]\), only an infimum of zero. Measuring the final velocity in the "
        r"initial frame would restore the Lorentz-composition constraint that produces "
        r"the supplied answer."
    ),
    "bdb3fc5f-9374-4e37-9af9-6036e3e29093": (
        r"Let \(\beta=v/c\). In the conductor's rest frame, "
        r"\(E'_0=\gamma(1-\beta)E_0\) and \(\omega'=\gamma(1-\beta)\omega\); a "
        r"stationary perfect conductor has "
        r"\(K'_y=2E'_0/(\mu_0c)\cos(\omega't')\). On the moving surface \(x=vt\), "
        r"\(t'=t/\gamma\), and the material sheet current transforms as "
        r"\(K_y=K'_y/\gamma\). This gives"
        r"\[\vec K(t)=\frac{2(1-\beta)E_0}{\mu_0c}"
        r"\cos\!\left[\omega(1-\beta)t\right]\hat{\mathbf y}.\]"
        r"Even the magnetic-jump convention gives "
        r"\(2E_0/[\mu_0c(1+\beta)]\cos[\omega(1-\beta)t]\,\hat{\mathbf y}\), so "
        r"the phase and amplitude in the supplied answer are not correct. For "
        r"completeness, the question should specify whether it asks for the material "
        r"surface current or the effective \(\mathbf n\times\mathbf H\) jump current."
    ),
    "1dba0086-bca8-4b25-8843-e6e428f9284d": (
        r"The supplied answer is correct if the Type B scale error is sampled "
        r"independently for each of the \(N\) readings, which contributes "
        r"\((a^2/12)/N\) to the variance of the mean. The question instead introduces "
        r"one random variable \(X_B\) and says that it is added to all results. This "
        r"naturally describes a common offset, for which the uncertainty is"
        r"\[\sqrt{\frac{s^2}{N}+\frac{a^2}{12}}.\]"
        r"The question must specify whether the scale error is common to all "
        r"measurements or independently resampled for each measurement."
    ),
    "2c46387b-36e6-43cb-9e73-dbf1341bfafd": (
        r"The value \(1.56\) follows if \(O\) is the fixed center of mass and \(P_2\) "
        r"and \(P_3\) are mirror-symmetric about \(OP\): "
        r"\(OP=(2/3)PP_2\cos(11.6^\circ)\), which gives \(OP_2\approx92.8L\). "
        r"These figure-eight choreography assumptions are not stated explicitly, and "
        r"the geometric data alone do not determine \(OP_2\). Review is needed to "
        r"decide whether ``eight-shaped orbit'' conventionally supplies the "
        r"center-of-mass and reflection symmetries required by the supplied answer."
    ),
    "85b4f862-d881-4a79-8c5d-3e927b486b71": (
        r"The titration gives \(0.003750\,\mathrm{mol}\) of iron atoms. Solving"
        r"\[x+y=0.225\,\mathrm{g},\qquad"
        r"\frac{x}{55.845}+\frac{2y}{159.687}=0.003750\]"
        r"gives approximately \(0.17316\,\mathrm{g}\) of \(\ce{Fe}\) and "
        r"\(0.05184\,\mathrm{g}\) of \(\ce{Fe2O3}\), or \(76.96\%\) and "
        r"\(23.04\%\), respectively. The supplied percentages do not follow from "
        r"standard molar masses; the question's data are otherwise sufficient."
    ),
    "cc251e69-6a32-4ed7-9315-beee80404866": (
        r"The supplied aniline is the expected product of a standard Curtius sequence "
        r"using ethyl chloroformate, \(\ce{Et3N}\), and \(\ce{NaN3}\), followed by "
        r"hydrolysis. The question instead specifies ethyl 2-chloroacetate consistently "
        r"by name, InChI, and SMILES. Ethyl 2-chloroacetate cannot form the mixed "
        r"anhydride or acyl azide required for the Curtius rearrangement; carboxylate "
        r"alkylation or recovery after hydrolysis is more plausible. The reaction "
        r"sequence as written therefore does not support the supplied aniline. "
        r"Replacing ethyl 2-chloroacetate with ethyl chloroformate appears to be the "
        r"minimal correction."
    ),
    "c982b5bc-0fb8-4f12-94fa-f05715209ead": (
        r"A tempting shortcut is to treat the \(50\,\mathrm{cm^3}\) absorbed by "
        r"\(\ce{Ca(OH)2}\) as \(\ce{CO2}\) and the difference "
        r"\(500-440=60\,\mathrm{cm^3}\) as condensed \(\ce{H2O}\). For an acyclic "
        r"saturated diol, \(\ce{C_{n}H_{2n+2}O2}\), the product ratio is "
        r"\((n+1)/n\); setting it equal to \(60/50\) gives \(n=5\) and hence the "
        r"supplied formula \(\ce{C5H12O2}\). Those quantities do not satisfy all of "
        r"the stated volume data. Combustion changes the hot-gas volume by "
        r"\((n+1)/2\) fuel volumes, so the initial and final volumes cannot both be "
        r"\(500\,\mathrm{cm^3}\) for any nonzero amount of diol. For \(n=5\), the "
        r"product data imply \(10\,\mathrm{cm^3}\) of fuel and an initial total "
        r"volume of \(470\,\mathrm{cm^3}\), not \(500\,\mathrm{cm^3}\). Comparing "
        r"gas volumes at \(300^\circ\mathrm C\) and room temperature without "
        r"additional conditions is also insufficiently specified."
    ),
    "b7337452-ab08-43e5-9f3b-419107415280": (
        r"The ionization-energy jumps after \(I_2\) for \(D\) and \(I_4\) for \(E\) "
        r"identify group 2 and group 14 behavior, respectively, giving formal highest "
        r"oxides \(DO\) and \(EO_2\). The numerical values correspond closely to "
        r"calcium and carbon, for which the supplied formula would be \(\ce{Ca2CO4}\), "
        r"not an ordinary compound. More generally, the condition \(D{:}E>1\) "
        r"excludes the common \(1{:}1\) product \(DEO_3\) but does not select "
        r"\(D_2EO_4\): formulas such as \(D_3EO_5\) and \(D_3E_2O_7\) also satisfy "
        r"formal charge balance and the stated ratio. Additional structural or "
        r"stoichiometric information is required."
    ),
    "f44b0903-1c29-44a5-9f9b-b6ccdd164e1c": (
        r"The supplied name appears to intend docosahexaenoic acid "
        r"(DHA, \(\mathrm{C}_{22}{:}6\)), but ``Decosahexanoic acid'' is not a "
        r"standard name. Assuming that glucosidase removes one hexose residue, the "
        r"parent mass is the phosphatidylglycerol mass plus \(\ce{C6H10O5}\). For two "
        r"identical \(\mathrm{C}_{n}{:}u\) acyl chains, the average parent mass is "
        r"approximately\[28.054n-4.032u+436.259.\]The stated molar mass of \(1026.5\) "
        r"gives no integer \(\mathrm{C}_{n}{:}u\) solution. A DHA-containing parent "
        r"would instead have a mass of approximately \(1028.6\) using exact masses or "
        r"\(1029.3\) using average masses. The data therefore do not uniquely support DHA."
    ),
    "3d762ba1-3958-453b-b34e-fe24c2822994": (
        r"The supplied expression is obtained if both constants are association "
        r"constants,\[K_s=\frac{[ES]}{[E][S]},\qquad "
        r"J_s=\frac{[SES]}{[ES][S]}.\]Under those definitions, "
        r"\(v=k[E]_0[S]/(K_s^{-1}+[S]+J_s[S]^2)\). If \(K_s\) and \(J_s\) are "
        r"instead dissociation constants, as is also common in enzyme kinetics, the "
        r"corresponding denominator is \(K_s+[S]+[S]^2/J_s\). The question calls both "
        r"quantities only ``equilibrium constants'' and does not define either "
        r"convention, so it does not determine a unique expression."
    ),
    "fcbd76ad-1b45-4f40-ad9d-d2bc5016f02c": (
        r"The viability experiment under amino-acid deprivation is attributed to "
        r"killed bacteria supplying additional nutrients, including amino acids, "
        r"through lysosomal recycling. The antioxidant-metabolite and "
        r"reactive-oxygen-species result is a separate finding, so ``increased "
        r"antioxidant availability'' conflates two conclusions. The amino-acid rescue "
        r"described in the question instead points directly to nutrient provision."
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_math(content: str) -> str:
    """Normalize source math without changing its mathematical content."""
    content = content.strip().replace("`", "")
    content = re.sub(r"\\{2,}", r"\\", content)
    content = content.replace(r"\_", "_").replace(r"\[", "[").replace(r"\]", "]")
    content = re.sub(r"\\([α-ωΑ-Ω])", r"\1", content)
    content = re.sub(r"√\s*\[([^\]]+)\]", r"\\sqrt{\1}", content)
    content = re.sub(r"√\s*\(([^)]+)\)", r"\\sqrt{\1}", content)
    content = re.sub(r"√\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", content)
    content = content.replace("√", r"\sqrt{}")
    for source, replacement in MATH_UNICODE.items():
        content = content.replace(source, replacement)
    for name in BARE_GREEK:
        content = re.sub(
            rf"(?<!\\)\b{name}(?=\b|_)",
            lambda _, command=name: rf"\{command}",
            content,
        )
    content = re.sub(
        r"(?<!\\)\b(sin|cos|tan|sinh|cosh|exp|ln)\b",
        lambda match: rf"\{match.group(1)}",
        content,
    )
    content = re.sub(r"\\(exp|sin|cos|sinh|cosh)!", r"\\\1\\!", content)
    content = content.replace(r"\lll", r"\ll")
    content = re.sub(
        r"(?<=\d)\s*cm\^\{?3\}?",
        lambda _: r"\,\mathrm{cm}^{3}",
        content,
    )
    content = re.sub(
        (
            r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|eta|theta|mu|pi|rho|"
            r"sigma|tau|chi|omega|Gamma|Delta|Omega|partial|ne|ll|gg|langle|rangle)"
            r"(?=[A-Za-z])"
        ),
        r"{\\\1}",
        content,
    )
    content = content.replace(r"{\ne}q", r"\neq")
    content = content.replace("%", r"\%")
    content = content.replace(r"k{B}", r"k_B")
    return content.strip()


def render_math_span(span: str, *, chemical_formulae: bool = False) -> str:
    if span.startswith("$"):
        display = False
        content = span[1:-1]
    else:
        stripped = span.lstrip("\\")
        display = stripped.startswith("[")
        content = re.sub(r"^\\+[\[(]", "", span)
        content = re.sub(r"\\+[\])]$", "", content)
    content = normalize_math(content)
    if chemical_formulae and re.search(r"\\(?:rightleftharpoons|rightarrow)", content):
        reaction = content.replace(r"\rightleftharpoons", "<=>")
        reaction = reaction.replace(r"\rightarrow", "->")
        reaction = re.sub(r"_\{(\d+)\}", r"\1", reaction)
        reaction = re.sub(r"_(\d+)", r"\1", reaction)
        reaction = re.sub(r"\s*(<=>|->)\s*", r" \1 ", reaction)
        reaction = re.sub(r"(?<!\^)\+(?!\})", " + ", reaction)
        return rf"\(\ce{{{reaction}}}\)"
    if chemical_formulae:
        formula = re.sub(r"_\{(\d+)\}", r"\1", content.replace(" ", ""))
        formula = re.sub(r"_(\d+)", r"\1", formula)
        if chemical_formula_math(formula) is not None:
            return rf"\(\ce{{{formula}}}\)"
    if display:
        return f"\\[{content}\\]"
    return f"\\({content}\\)"


def escape_prose(text: str) -> str:
    escaped = re.sub(
        r"[\\&%$#_{}~^<>]",
        lambda match: PROSE_ESCAPES[match.group()],
        text,
    )
    for source, replacement in TEXT_UNICODE.items():
        escaped = escaped.replace(source, replacement)
    return escaped


def render_identifier(kind: str, value: str) -> str:
    value = value.strip().replace(r"\[", "[").replace(r"\]", "]")
    delimiter = next((candidate for candidate in "!|;:+" if candidate not in value), None)
    if delimiter is None:
        raise ValueError(f"no safe listings delimiter for {kind}: {value}")
    label = "InChI" if kind == "INCHI" else kind
    return (
        rf"\textsc{{{label}}}: "
        rf"\lstinline[breaklines=true]{delimiter}{value}{delimiter}"
    )


def mark_plain_formulae(text: str, stash: Any) -> str:
    """Typeset un-delimited molecular formulae that contain a numeral."""

    def replace(match: re.Match[str]) -> str:
        candidate = match.group()
        if not re.search(r"\d", candidate):
            return candidate
        rendered = chemical_formula_math(candidate)
        if rendered is None:
            return candidate
        return stash(rf"\(\ce{{{rendered}}}\)")

    text = re.sub(
        r"(?<![A-Za-z0-9/])Fe2\+",
        lambda _: stash(r"\(\ce{Fe^2+}\)"),
        text,
    )
    return re.sub(
        (
            r"(?<![A-Za-z0-9/])[A-Z][A-Za-z0-9]*"
            r"(?:\([A-Za-z0-9]+\)[A-Za-z0-9]*)*(?![A-Za-z0-9])"
        ),
        replace,
        text,
    )


def render_markdown_table(block: str, *, chemical_formulae: bool = False) -> str:
    rows: list[list[str]] = []
    for line in block.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        if cells and any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rendered_rows = []
    for row in rows:
        padded = row + [""] * (width - len(row))
        rendered_rows.append(
            " & ".join(render_prose(cell, chemical_formulae=chemical_formulae) for cell in padded)
            + r" \\"
        )
    columns = "l" * width
    return (
        "\n".join(
            [
                r"\begin{center}",
                rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
                *rendered_rows,
                r"\end{tabular}",
                r"\end{center}",
            ]
        )
        + "\n"
    )


def render_markdown_bullets(block: str, *, chemical_formulae: bool = False) -> str:
    items = [line[2:].strip() for line in block.splitlines() if line.startswith("- ")]
    return (
        "\n".join(
            [
                r"\begin{itemize}[leftmargin=2em,itemsep=0.2em]",
                *(
                    rf"\item {render_prose(item, chemical_formulae=chemical_formulae)}"
                    for item in items
                ),
                r"\end{itemize}",
            ]
        )
        + "\n"
    )


def render_prose(text: str, *, chemical_formulae: bool = False) -> str:
    """Render mixed Markdown-ish prose and LaTeX math as ASCII LaTeX."""
    text = html.unescape(text).replace("`", "")
    text = re.sub(
        r"\\\[(\d+\+\d+)\\\]",
        lambda match: rf"\([{match.group(1)}]\)",
        text,
    )
    text = re.sub(
        r"\bKsp\s*=\s*\\\(\s*([^)]*?)\s*\\\)",
        lambda match: rf"\(K_{{sp}}={match.group(1)}\)",
        text,
    )
    text = re.sub(
        r"ω\(([^)]*)\)\s*=\s*([0-9.]+)%",
        lambda match: rf"\(\omega({match.group(1)})={match.group(2)}%\)",
        text,
    )
    text = re.sub(
        r"ω\(([^)]*)\)",
        lambda match: rf"\(\omega({match.group(1)})\)",
        text,
    )
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*g/cm³",
        lambda match: rf"\({match.group(1)}\,\mathrm{{g\,cm^{{-3}}}}\)",
        text,
    )
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*dm³",
        lambda match: rf"\({match.group(1)}\,\mathrm{{dm}}^3\)",
        text,
    )
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*g(?:/mol|\s+mol⁻¹)",
        lambda match: rf"\({match.group(1)}\,\mathrm{{g\,mol^{{-1}}}}\)",
        text,
    )
    text = re.sub(
        r"\b1-?H NMR",
        lambda _: r"\({}^{1}\mathrm{H}\) NMR",
        text,
    )
    text = text.replace(
        r"a) \[\( S]_0 >> [E]_0 \)",
        r"a) \([S]_0 \gg [E]_0\)",
    )
    text = text.replace("kJ/mol", r"\(\mathrm{kJ\,mol^{-1}}\)")
    text = re.sub(
        r"(\d+(?:\.\d+)?)°C",
        lambda match: rf"\({match.group(1)}^\circ\mathrm{{C}}\)",
        text,
    )
    text = re.sub(
        r"(?m)^(Element [DE]:)\s*(I[₀-₉].*)$",
        lambda match: (
            f"{match.group(1)} "
            rf"\({normalize_math(match.group(2))}\)"
        ),
        text,
    )
    text = text.replace("〈.〉", r"\(\langle\,\cdot\,\rangle\)")
    text = text.replace("𝑓(𝑃)", r"\(f(P)\)")
    text = text.replace("⁻¹", r"\({}^{-1}\)")
    text = re.sub(
        r"\b(kg|mol|m|s|cm|dm)−(\d+)",
        lambda match: rf"{match.group(1)}\(^{{-{match.group(2)}}}\)",
        text,
    )
    text = re.sub(
        r"([A-Za-z])¨",
        lambda match: rf"\(\ddot{{{match.group(1)}}}\)",
        text,
    )
    text = re.sub(
        r"([A-Za-z])˙",
        lambda match: rf"\(\dot{{{match.group(1)}}}\)",
        text,
    )
    text = re.sub(r"√\[([^\]]+)\]", lambda match: rf"\(\sqrt{{{match.group(1)}}}\)", text)
    text = re.sub(r"(?m)\\+\s*$", "", text)

    tokens: list[str] = []

    def stash(rendered: str) -> str:
        marker = f"@@LATEX{len(tokens):04d}@@"
        tokens.append(rendered)
        return marker

    text = MARKDOWN_TABLE.sub(
        lambda match: stash(
            render_markdown_table(match.group(), chemical_formulae=chemical_formulae)
        ),
        text,
    )
    text = MARKDOWN_BULLETS.sub(
        lambda match: stash(
            render_markdown_bullets(match.group(), chemical_formulae=chemical_formulae)
        ),
        text,
    )

    def replace_tag(match: re.Match[str]) -> str:
        kind = match.group(1).upper()
        value = match.group(2).strip()
        if kind in {"SMILES", "INCHI"}:
            return stash(render_identifier(kind, value))
        return value.replace(r"\[", "[").replace(r"\]", "]")

    # Protect structure tags before scanning for math: bracket atoms such as
    # ``[nH]`` and ``[C@@]`` are chemical syntax, not display delimiters.
    text = MALFORMED_TAG.sub(replace_tag, text)
    if chemical_formulae:
        text = re.sub(
            r"\\+\[([A-Z][A-Za-z0-9]*)\\+\](\d*)([+-])",
            lambda match: stash(rf"\(\ce{{{match.group(1)}^{match.group(2)}{match.group(3)}}}\)"),
            text,
        )
    # Escaped bracketed element/complex notation is also literal chemistry.
    text = re.sub(r"\\+\[([A-Z][A-Za-z0-9@+()\-]*)\\+\]", r"[\1]", text)
    if chemical_formulae:
        text = mark_parenthesized_formulae(text)
    text = MATH_SPAN.sub(
        lambda match: stash(render_math_span(match.group(), chemical_formulae=chemical_formulae)),
        text,
    )
    text = re.sub(r"\bmu_0\b", lambda _: stash(r"\(\mu_0\)"), text)
    text = re.sub(
        r"\bmol\s+L\s*\^?\s*\{?-?1\}?",
        lambda _: stash(r"\(\mathrm{mol\,L^{-1}}\)"),
        text,
    )

    def replace_quantity(match: re.Match[str]) -> str:
        if match.group(2).startswith(("cm^", "dm^")):
            base = match.group(2)[:2]
            unit = rf"\mathrm{{{base}}}^{{3}}"
        else:
            unit = rf"\mathrm{{{match.group(2)}}}"
        return stash(rf"\({match.group(1)}\,{unit}\)")

    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*([cd]m\^\{?3\}?|mL|mg|g|M|ppm)(?![A-Za-z])",
        replace_quantity,
        text,
    )
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)%",
        lambda match: stash(rf"\({match.group(1)}\%\)"),
        text,
    )
    if chemical_formulae:
        text = mark_plain_formulae(text, stash)
        text = re.sub(
            r"(?<![A-Za-z0-9])([A-Z])(\d+)(?![A-Za-z0-9])",
            lambda match: stash(rf"\({match.group(1)}_{{{match.group(2)}}}\)"),
            text,
        )
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    for command, symbol in BARE_GREEK.items():
        text = re.sub(rf"\\+{command}\b", symbol, text)
    text = text.replace("**", "").replace("*", "")
    text = escape_prose(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    for index, rendered in enumerate(tokens):
        text = text.replace(f"@@LATEX{index:04d}@@", rendered)
    return text.strip()


def render_problem(text: str, *, chemical_formulae: bool = False) -> str:
    """Render a problem and center paragraphs that contain only one equation."""
    rendered = render_prose(text, chemical_formulae=chemical_formulae)
    return re.sub(
        r"(?m)^[ \t]*\\\(((?:(?!\\\)).)+)\\\)([.,]?)[ \t]*$",
        lambda match: rf"\[{match.group(1)}{match.group(2)}\]",
        rendered,
    )


def render_quantity(text: str) -> str | None:
    match = re.fullmatch(
        r"\s*([+\-]?\s*\d+(?:\.\d+)?)\s*(micrograms|mg|g|M|V|J|K)?\.?\s*",
        text.replace(r"\-", "-"),
        re.IGNORECASE,
    )
    if not match:
        return None
    number = match.group(1).replace(" ", "")
    unit = match.group(2)
    if not unit:
        return rf"\({number}\)"
    unit_tex = r"\mu\mathrm{g}" if unit.lower() == "micrograms" else rf"\mathrm{{{unit}}}"
    return rf"\({number}\,{unit_tex}\)"


def looks_like_bare_math(text: str) -> bool:
    stripped = text.strip().strip("`")
    return bool(
        re.match(r"^(?:[A-Za-z]+(?:_[A-Za-z0-9]+)?|[α-ωΑ-Ω])\s*=", stripped)
        or any(char in stripped for char in MATH_UNICODE)
        or ("=" in stripped and not re.search(r"\b(?:is|where|with)\b", stripped, re.I))
    )


def render_choice(text: str, *, chemical_formulae: bool = False) -> str:
    decoded = html.unescape(text).strip().strip("`").strip()
    # Choice lists need inline math. Display delimiters center an expression
    # independently of its choice label and reflow differently inside a box.
    decoded = re.sub(r"\\\[(.*?)\\\]", r"\\(\1\\)", decoded, flags=re.DOTALL)
    # In compound-name choices, escaped square brackets delimit coordination
    # complexes; a whole-choice display formula instead begins with ``\[``.
    if not decoded.startswith(r"\["):
        decoded = decoded.replace(r"\[", "[").replace(r"\]", "]")
    quantity = render_quantity(decoded)
    if quantity is not None:
        return quantity
    if not MATH_SPAN.search(decoded) and not TAG.search(decoded) and looks_like_bare_math(decoded):
        return rf"\({normalize_math(decoded)}\)"
    return render_prose(decoded, chemical_formulae=chemical_formulae)


def chemical_formula_math(formula: str) -> str | None:
    if re.fullmatch(r"[IVX]+", formula):
        return None
    if formula.startswith("["):
        return None
    pieces: list[str] = []
    element_count = 0
    position = 0
    delimiters: list[str] = []
    while position < len(formula):
        element = re.match(r"[A-Z][a-z]?", formula[position:])
        if element:
            symbol = element.group()
            if symbol not in ELEMENTS:
                return None
            pieces.append(symbol)
            element_count += 1
            position += len(symbol)
            continue
        number = re.match(r"\d+", formula[position:])
        if number:
            value = number.group()
            pieces.append(value)
            position += len(value)
            continue
        char = formula[position]
        if char in "([":
            delimiters.append(char)
            pieces.append(char)
            position += 1
            continue
        if char in ")]":
            expected = "(" if char == ")" else "["
            if not delimiters or delimiters.pop() != expected:
                return None
            pieces.append(char)
            position += 1
            continue
        return None
    if delimiters:
        return None
    if element_count < 1:
        return None
    if element_count == 1 and formula not in {
        "H2",
        "N2",
        "O2",
        "O3",
        "F2",
        "P4",
        "S8",
        "Cl2",
        "Br2",
        "I2",
        "C60",
    }:
        return None
    return "".join(pieces)


def mark_parenthesized_formulae(text: str) -> str:
    """Wrap balanced parenthesized chemical formulae in math delimiters."""
    replacements: list[tuple[int, int, str]] = []
    stack: list[int] = []
    for index, char in enumerate(text):
        if char == "(":
            stack.append(index)
        elif char == ")" and stack:
            start = stack.pop()
            if stack:
                continue
            formula = text[start + 1 : index]
            if len(formula) > 50 or not re.fullmatch(r"[A-Za-z0-9()[\]]+", formula):
                continue
            rendered = chemical_formula_math(formula)
            if rendered is not None:
                replacements.append((start, index + 1, rf"(\(\ce{{{rendered}}}\))"))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def preamble(title: str) -> list[str]:
    return [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.85in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage[version=4]{mhchem}",
        r"\usepackage{enumitem}",
        r"\usepackage{listings}",
        r"\usepackage{varwidth}",
        r"\usepackage{xcolor}",
        r"\usepackage[hidelinks]{hyperref}",
        (
            r"\lstset{basicstyle=\ttfamily\small,breaklines=true,"
            r"columns=fullflexible,keepspaces=true}"
        ),
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.55em}",
        r"\setlength{\emergencystretch}{3em}",
        r"\allowdisplaybreaks",
        r"\raggedbottom",
        r"\setlength{\fboxsep}{4pt}",
        r"\newsavebox{\correctanswerbox}",
        (
            r"\newenvironment{correctanswer}"
            r"{\begin{lrbox}{\correctanswerbox}"
            r"\begin{varwidth}{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}}"
            r"{\end{varwidth}\end{lrbox}\fbox{\usebox{\correctanswerbox}}}"
        ),
        r"\newcommand{\choicelabel}[1]{\makebox[2em][l]{#1.}}",
        r"\newcommand{\hfitem}[1]{\textbf{HF item #1}}",
        rf"\title{{{title}}}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]


def build_hf_index(hf_rows: list[dict[str, Any]]) -> dict[str, int]:
    index = {str(row["task_group_id"]): number for number, row in enumerate(hf_rows)}
    if len(index) != len(hf_rows):
        raise ValueError("Olympiad task_group_id values must be unique")
    return index


def write_problematic(
    output: Path,
    review_records: list[dict[str, Any]],
    hf_index: dict[str, int],
) -> None:
    records = sorted(
        review_records,
        key=lambda record: (SUBJECTS.index(record["subject"]), hf_index[record["task_group_id"]]),
    )
    record_ids = {record["task_group_id"] for record in records}
    if record_ids != set(REVIEW_ANALYSIS_LATEX):
        raise ValueError("consultant analysis must cover every review item exactly once")
    if not record_ids > INCORRECT_REVIEW_IDS:
        raise ValueError("incorrect-review classification contains an unknown item")

    lines = preamble(r"FrontierScience Olympiad\\Questions for External Review")
    lines.extend(
        [
            (
                "This document contains 17 questions for independent subject-matter review. For "
                "each item, the answer supplied with the original dataset is reproduced without "
                "correction, followed by a preliminary analysis of the issue. The alphanumeric "
                "identifier and HF item number are retained to make each question easy to trace. "
                "HF item numbers are zero-based row indices in the pinned Olympiad split."
            ),
            "",
            r"For each question, please determine:",
            r"\begin{enumerate}[leftmargin=2em]",
            r"\item Does the question as written determine a unique, scientifically valid answer?",
            r"\item Is the supplied answer correct?",
            (
                r"\item If not, what answer or minimal revision would make the item correct and "
                r"unambiguous?"
            ),
            r"\end{enumerate}",
        ]
    )
    removed_instructions = 0
    for subject in SUBJECTS:
        subset = [record for record in records if record["subject"] == subject]
        if not subset:
            continue
        lines.extend([r"\clearpage", rf"\section{{{subject.title()}}}", ""])
        groups = (
            (
                "Incorrect supplied answers",
                [record for record in subset if record["task_group_id"] in INCORRECT_REVIEW_IDS],
            ),
            (
                "Ambiguous or internally inconsistent questions",
                [
                    record
                    for record in subset
                    if record["task_group_id"] not in INCORRECT_REVIEW_IDS
                ],
            ),
        )
        for heading, group in groups:
            lines.extend([rf"\subsection{{{heading} ({len(group)})}}", ""])
            if not group:
                lines.extend(
                    [
                        (
                            f"No {subject} questions in this review set were classified in this "
                            "category."
                        ),
                        "",
                    ]
                )
                continue
            for record in group:
                item_id = record["task_group_id"]
                question, substitutions = ANSWER_INSTRUCTION.subn("", record["question"])
                removed_instructions += substitutions
                lines.extend(
                    [
                        rf"\subsubsection{{HF item {hf_index[item_id]}}}",
                        rf"\texttt{{{item_id}}}",
                        "",
                        r"\paragraph*{Question}",
                        render_problem(question, chemical_formulae=subject == "chemistry"),
                        "",
                        r"\paragraph*{Supplied answer}",
                        REVIEW_REFERENCE_LATEX.get(
                            item_id,
                            render_choice(
                                record["supplied_reference_answer"],
                                chemical_formulae=subject == "chemistry",
                            ),
                        ),
                        "",
                        r"\paragraph*{Preliminary analysis}",
                        REVIEW_ANALYSIS_LATEX[item_id],
                        "",
                    ]
                )

    if removed_instructions != len(records):
        raise ValueError(
            "removed answer-format instruction from "
            f"{removed_instructions}/{len(records)} review records"
        )
    lines.append(r"\end{document}")
    output.write_text("\n".join(lines) + "\n")


def write_verified(
    output: Path,
    records: list[dict[str, Any]],
    hf_index: dict[str, int],
) -> None:
    records = sorted(
        records,
        key=lambda record: (SUBJECTS.index(record["subject"]), hf_index[record["task_group_id"]]),
    )
    lines = preamble("FrontierScience Olympiad: Verified Multiple-Choice Teacher Copy")
    lines.extend(
        [
            (
                "This teacher copy contains the 83 questions without an active source-answer or "
                "source-stem concern. Questions are grouped by subject and ordered by their "
                "zero-based Hugging Face row index. Correct choices are enclosed in boxes."
            ),
            "",
        ]
    )
    removed_instructions = 0
    for subject_index, subject in enumerate(SUBJECTS):
        subset = [record for record in records if record["subject"] == subject]
        if not subset:
            continue
        if subject_index:
            lines.append(r"\clearpage")
        lines.extend([rf"\section{{{subject.title()} ({len(subset)})}}", ""])
        for record in subset:
            item_id = record["task_group_id"]
            problem, substitutions = ANSWER_INSTRUCTION.subn("", record["problem"])
            removed_instructions += substitutions
            lines.extend(
                [
                    rf"\subsection{{HF item {hf_index[item_id]}}}",
                    rf"\texttt{{{item_id}}}",
                    "",
                    render_problem(problem, chemical_formulae=subject == "chemistry"),
                    "",
                    r"\begin{enumerate}[label={},leftmargin=0pt,itemsep=0.55em]",
                ]
            )
            for choice_index, choice in enumerate(record["choices"]):
                rendered = render_choice(choice, chemical_formulae=subject == "chemistry")
                if choice_index == record["correct_choice_index"]:
                    lines.extend(
                        [
                            r"\item",
                            r"\hspace*{\dimexpr-\fboxsep-\fboxrule\relax}%",
                            r"\begin{correctanswer}",
                            rf"\hangindent=2em\hangafter=1\choicelabel"
                            rf"{{{CHOICE_LABELS[choice_index]}}}{rendered}",
                            r"\end{correctanswer}",
                        ]
                    )
                else:
                    lines.append(
                        rf"\item \hangindent=2em\hangafter=1\choicelabel"
                        rf"{{{CHOICE_LABELS[choice_index]}}}{rendered}"
                    )
            lines.extend([r"\end{enumerate}", ""])

    if removed_instructions != len(records):
        raise ValueError(
            f"removed answer-format instruction from {removed_instructions}/{len(records)} records"
        )

    lines.extend([r"\clearpage", r"\section*{Compact answer key}"])
    for subject in SUBJECTS:
        subset = [record for record in records if record["subject"] == subject]
        if not subset:
            continue
        entries = [
            f"{hf_index[record['task_group_id']]}--{record['correct_choice_label']}"
            for record in subset
        ]
        lines.extend(
            [
                rf"\textbf{{{subject.title()}:}} " + ", ".join(entries) + ".",
                "",
            ]
        )
    lines.append(r"\end{document}")
    output.write_text("\n".join(lines) + "\n")


def assert_ascii_and_structure(
    path: Path,
    expected_questions: int,
    *,
    question_heading: str = r"\subsection{",
) -> None:
    text = path.read_text()
    non_ascii = sorted(set(char for char in text if ord(char) > 127))
    if non_ascii:
        raise ValueError(f"{path} contains non-ASCII characters: {non_ascii}")
    if text.count(question_heading) != expected_questions:
        raise ValueError(f"{path} has an unexpected question count")
    if text.count(r"\begin{document}") != 1 or text.count(r"\end{document}") != 1:
        raise ValueError(f"{path} has unbalanced document delimiters")
    if text.count("{") != text.count("}"):
        raise ValueError(f"{path} has unbalanced braces")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-items", type=Path, required=True)
    parser.add_argument("--verified-items", type=Path, required=True)
    parser.add_argument("--review-items", type=Path, required=True)
    parser.add_argument("--hf-input", type=Path, required=True)
    parser.add_argument("--problematic-output", type=Path, required=True)
    parser.add_argument("--verified-output", type=Path, required=True)
    args = parser.parse_args()

    all_records = load_jsonl(args.all_items)
    verified_records = load_jsonl(args.verified_items)
    review_records = load_jsonl(args.review_items)
    hf_rows = load_jsonl(args.hf_input)
    hf_index = build_hf_index(hf_rows)

    if len(all_records) != 100 or len(verified_records) != 83 or len(review_records) != 17:
        raise ValueError("expected current 100/83/17 artifact counts")
    if {record["task_group_id"] for record in all_records} != set(hf_index):
        raise ValueError("materialized and Hugging Face item IDs do not match")

    args.problematic_output.parent.mkdir(parents=True, exist_ok=True)
    write_problematic(args.problematic_output, review_records, hf_index)
    write_verified(args.verified_output, verified_records, hf_index)
    assert_ascii_and_structure(
        args.problematic_output,
        17,
        question_heading=r"\subsubsection{HF item ",
    )
    assert_ascii_and_structure(args.verified_output, 83)


if __name__ == "__main__":
    main()
