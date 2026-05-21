from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
OUTPUT = ROOT / "docs" / "FIFA_World_Cup_Predictor_Capstone.pptx"

SLIDE_W = 12192000
SLIDE_H = 6858000

COLORS = {
    "pitch": "0B5D3B",
    "dark": "07321F",
    "mint": "78D6A3",
    "gold": "D6B35A",
    "white": "FFFFFF",
    "ink": "102017",
    "muted": "50675C",
    "panel": "F7FBF6",
}


def emu(inches):
    return int(inches * 914400)


def safe(text):
    return escape(str(text))


def shape_rect(x, y, w, h, fill, line=None):
    ln = ""
    if line:
        ln = f"""
        <a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>"""
    else:
        ln = "<a:ln><a:noFill/></a:ln>"
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="1" name="Rectangle"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
        {ln}
      </p:spPr>
      <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
    </p:sp>"""


def text_box(text, x, y, w, h, size=28, color="FFFFFF", bold=False, align="l"):
    b = ' b="1"' if bold else ""
    paragraphs = str(text).split("\n")
    parts = []
    for para in paragraphs:
        parts.append(
            f"""
            <a:p>
              <a:pPr algn="{align}"/>
              <a:r>
                <a:rPr lang="en-US" sz="{size * 100}"{b}>
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                  <a:latin typeface="Aptos"/>
                </a:rPr>
                <a:t>{safe(para)}</a:t>
              </a:r>
            </a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="2" name="Text"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/><a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" anchor="t"/>
        <a:lstStyle/>
        {''.join(parts)}
      </p:txBody>
    </p:sp>"""


def bullet_list(items, x, y, w, h, size=21, color="102017"):
    parts = []
    for item in items:
        parts.append(
            f"""
            <a:p>
              <a:pPr marL="285750" indent="-171450"><a:buChar char="•"/></a:pPr>
              <a:r>
                <a:rPr lang="en-US" sz="{size * 100}">
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                  <a:latin typeface="Aptos"/>
                </a:rPr>
                <a:t>{safe(item)}</a:t>
              </a:r>
            </a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="3" name="Bullets"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/><a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{''.join(parts)}</p:txBody>
    </p:sp>"""


def image_box(rid, x, y, w, h):
    return f"""
    <p:pic>
      <p:nvPicPr><p:cNvPr id="4" name="Screenshot"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>"""


def bars(data, x, y, w, h):
    max_value = max(value for _, value in data)
    gap = emu(0.12)
    bar_h = int((h - gap * (len(data) - 1)) / len(data))
    out = []
    for i, (label, value) in enumerate(data):
        yy = y + i * (bar_h + gap)
        bw = int(w * value / max_value)
        out.append(shape_rect(x, yy, bw, bar_h, COLORS["pitch"]))
        out.append(text_box(label, x + emu(0.1), yy + emu(0.03), emu(2.4), bar_h, 14, COLORS["white"], True))
        out.append(text_box(f"{value:.1%}", x + bw + emu(0.12), yy + emu(0.03), emu(1.0), bar_h, 14, COLORS["ink"], True))
    return "".join(out)


def slide_xml(content, bg="FFFFFF"):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{bg}"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="0" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {content}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def slide_rels(image_name=None):
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    if image_name:
        rels.append(
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_name}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def title_slide():
    return slide_xml(
        shape_rect(0, 0, SLIDE_W, SLIDE_H, COLORS["dark"])
        + shape_rect(emu(0.45), emu(0.45), emu(12.45), emu(6.6), COLORS["pitch"], COLORS["gold"])
        + text_box("Predicting FIFA World Cup Outcomes", emu(0.9), emu(1.05), emu(9.2), emu(1.6), 44, COLORS["white"], True)
        + text_box("Using player attributes, team composition, rankings, and match-level analysis", emu(0.95), emu(2.55), emu(8.8), emu(0.8), 24, "DDEFE5")
        + text_box("Senior Capstone | Interactive Streamlit predictor and 2026 simulator", emu(0.95), emu(5.85), emu(8.7), emu(0.5), 18, COLORS["mint"], True),
        COLORS["dark"],
    )


def standard_slide(title, body, proof=None):
    proof_text = "" if not proof else text_box(proof, emu(8.55), emu(1.25), emu(3.7), emu(4.9), 22, COLORS["ink"], True)
    proof_panel = "" if not proof else shape_rect(emu(8.25), emu(1.0), emu(4.25), emu(5.35), "EAF5EE", COLORS["mint"])
    return slide_xml(
        text_box(title, emu(0.65), emu(0.45), emu(10.8), emu(0.65), 30, COLORS["pitch"], True)
        + shape_rect(emu(0.65), emu(1.25), emu(7.05), emu(5.25), "F7FBF6", "C9DDD1")
        + bullet_list(body, emu(0.9), emu(1.55), emu(6.5), emu(4.65), 20, COLORS["ink"])
        + proof_panel
        + proof_text
    )


def image_slide(title, image_name, caption):
    return slide_xml(
        text_box(title, emu(0.65), emu(0.42), emu(11.2), emu(0.65), 28, COLORS["pitch"], True)
        + image_box("rId2", emu(0.65), emu(1.18), emu(8.1), emu(5.45))
        + shape_rect(emu(9.05), emu(1.18), emu(3.5), emu(5.45), "F7FBF6", "C9DDD1")
        + text_box(caption, emu(9.3), emu(1.55), emu(3.0), emu(4.7), 20, COLORS["ink"], True)
    )


def build():
    slides = [
        (title_slide(), None),
        (
            standard_slide(
                "Research Question",
                [
                    "Can player attributes and team composition explain international match outcomes?",
                    "Can the same model power a forward-looking 2026 World Cup simulation?",
                    "How do multiple model families compare on the same connected feature set?",
                ],
                "Goal: move from raw football data to an interactive, reproducible prediction system.",
            ),
            None,
        ),
        (
            standard_slide(
                "Data Foundation",
                [
                    "FIFA18 player attributes provide player-level strength and position signals.",
                    "International match results provide outcomes and goal difference.",
                    "FIFA rankings add external performance context.",
                    "World Cup history supports tournament-success modeling.",
                    "2026 group data powers the simulator.",
                ],
                "Key limitation: possession, shots, fouls, and exact 2018 squad files are not present locally.",
            ),
            None,
        ),
        (
            standard_slide(
                "Feature Engineering",
                [
                    "Player data is aggregated into national-team features.",
                    "Features include overall, attack, midfield, defense, goalkeeper, age, balance, player count, elite players, rank, and points.",
                    "Match rows use differences between Team A and Team B for consistency.",
                ],
                "Pipeline: players -> team features -> match differences -> model targets.",
            ),
            None,
        ),
        (
            standard_slide(
                "Modeling Tests",
                [
                    "Match outcome: Logistic Regression, Random Forest, SVM, Naive Bayes, MLP.",
                    "Goal difference: Linear, Ridge, Lasso, Random Forest, SVR, MLP.",
                    "Tournament success: Logistic Regression, Decision Tree, Random Forest, MLP.",
                ],
                "Best match model: SVM\nAccuracy: 72.6%\nTraining rows: 4,951\nTournament rows: 126",
            ),
            None,
        ),
        (
            slide_xml(
                text_box("Match Model Results", emu(0.65), emu(0.42), emu(11.2), emu(0.65), 30, COLORS["pitch"], True)
                + shape_rect(emu(0.85), emu(1.35), emu(6.8), emu(4.9), "F7FBF6", "C9DDD1")
                + bars(
                    [
                        ("SVM", 0.7255),
                        ("Naive Bayes", 0.7185),
                        ("Logistic", 0.7104),
                        ("MLP", 0.6801),
                        ("Random Forest", 0.6690),
                        ("Baseline", 0.5257),
                    ],
                    emu(1.1),
                    emu(1.7),
                    emu(5.2),
                    emu(3.9),
                )
                + shape_rect(emu(8.15), emu(1.35), emu(3.65), emu(4.9), "EAF5EE", COLORS["mint"])
                + text_box("The SVM performs best on the held-out test set, improving about 20 percentage points over the dummy baseline.", emu(8.45), emu(1.75), emu(3.05), emu(3.7), 22, COLORS["ink"], True)
            ),
            None,
        ),
        (
            image_slide(
                "Interactive Match Predictor",
                "match-predictor.png",
                "The app lets users select two teams and generate real-time win probabilities from the saved model artifacts.",
            ),
            "match-predictor.png",
        ),
        (
            image_slide(
                "2026 World Cup Simulation",
                "world-cup-simulator.png",
                "The simulator runs repeated tournaments across the 48-team, 12-group format and summarizes advancement and champion probabilities.",
            ),
            "world-cup-simulator.png",
        ),
        (
            image_slide(
                "Model Report Evidence",
                "model-report.png",
                "The report tab documents the full pipeline, model comparisons, confusion matrix, feature importance, and known data gaps.",
            ),
            "model-report.png",
        ),
        (
            standard_slide(
                "Conclusion",
                [
                    "The project now runs end to end from raw data to an interactive dashboard.",
                    "The model performs meaningfully better than baseline.",
                    "The 2026 simulation demonstrates how probability outputs can support tournament forecasting.",
                    "Future work should add exact squad rosters and richer in-game match statistics.",
                ],
                "Final deliverables: code pipeline, Streamlit app, report draft, screenshots, and presentation deck.",
            ),
            None,
        ),
    ]

    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as ppt:
        ppt.writestr("[Content_Types].xml", content_types(len(slides)))
        ppt.writestr("_rels/.rels", package_rels())
        ppt.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        ppt.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        ppt.writestr("ppt/theme/theme1.xml", theme_xml())
        ppt.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        ppt.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        ppt.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        ppt.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())

        media_written = set()
        for idx, (xml, image_name) in enumerate(slides, start=1):
            ppt.writestr(f"ppt/slides/slide{idx}.xml", xml)
            ppt.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels(image_name))
            if image_name and image_name not in media_written:
                ppt.write(ASSETS / image_name, f"ppt/media/{image_name}")
                media_written.add(image_name)


def content_types(slide_count):
    slides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  {slides}
</Types>"""


def package_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""


def presentation_xml(slide_count):
    slide_ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, slide_count + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{slide_count + 1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(slide_count):
    rels = [
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, slide_count + 1)
    ]
    rels.append(
        f'<Relationship Id="rId{slide_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    )
    rels.append(
        f'<Relationship Id="rId{slide_count + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def theme_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="FIFA Predictor">
  <a:themeElements>
    <a:clrScheme name="FIFA"><a:dk1><a:srgbClr val="102017"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="0B5D3B"/></a:dk2><a:lt2><a:srgbClr val="F7FBF6"/></a:lt2><a:accent1><a:srgbClr val="0B5D3B"/></a:accent1><a:accent2><a:srgbClr val="D6B35A"/></a:accent2><a:accent3><a:srgbClr val="78D6A3"/></a:accent3><a:accent4><a:srgbClr val="50675C"/></a:accent4><a:accent5><a:srgbClr val="EAF5EE"/></a:accent5><a:accent6><a:srgbClr val="07321F"/></a:accent6><a:hlink><a:srgbClr val="0B5D3B"/></a:hlink><a:folHlink><a:srgbClr val="50675C"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Simple"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def slide_master_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_master_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def slide_layout_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


if __name__ == "__main__":
    build()
    print(OUTPUT)
