from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "Assets"
REPORT_ASSETS = ASSETS / "report_assets"
OUT = ASSETS / "RL_Final_Presentation_ContentOnly.pptx"

EMU_PER_INCH = 914400
SLIDE_W = 13.333333 * EMU_PER_INCH
SLIDE_H = 7.5 * EMU_PER_INCH


slides = [
    {
        "title": "การปรับค่า PID ด้วย Deep Reinforcement Learning สำหรับ 3-RRS Ball Balancer",
        "bullets": [
            "หัวข้อตาม requirement: project overview, scopes, background knowledge, problem formulation, techniques, simulation, result และ analysis",
            "ผล PPO สุดท้ายใช้ run: pid_tuning_multi_12",
            "ผล DDPG สุดท้ายใช้ run: DDPG_14",
        ],
    },
    {
        "title": "Project Overview",
        "bullets": [
            "โครงงานศึกษาการใช้ DRL เพื่อปรับค่า PID gains สำหรับระบบลูกบอลทรงตัวที่จำลองใน Unity",
            "ระบบกลไกเป็นแพลตฟอร์มขนาน 3-RRS แบบ 3 องศาอิสระ",
            "agent ส่งออกค่า PID gains แบบต่อเนื่อง ได้แก่ Kp, Ki และ Kd",
            "inner PID controller ใช้ค่า gains เหล่านี้สร้างคำสั่งควบคุมแพลตฟอร์ม",
        ],
    },
    {
        "title": "Scope",
        "bullets": [
            "ขอบเขตที่ทำ: Unity simulation, rigid-body ball motion, platform behavior, ML-Agents communication, PPO training และ custom DDPG training",
            "algorithm ที่เปรียบเทียบ: PPO และ DDPG-FC-350-E-PID",
            "ข้อมูลประเมินหลักมาจาก TensorBoard logs และ training_status.json ของ run สุดท้าย",
            "สิ่งที่ยังไม่รวม: การทดสอบกับ hardware จริง และ sim-to-real transfer",
        ],
    },
    {
        "title": "Background Knowledge",
        "bullets": [
            "PID control ตีความง่ายและใช้ได้ดีในงานควบคุมหลายแบบ แต่ fixed gains อาจไม่เหมาะกับระบบ nonlinear ทุกสถานะ",
            "DRL สามารถเรียนรู้ policy สำหรับปรับ PID gains จากสถานะที่สังเกตได้",
            "PPO เป็น on-policy actor-critic ที่ใช้ clipped policy update",
            "DDPG เป็น off-policy deterministic actor-critic สำหรับ continuous action space",
        ],
    },
    {
        "title": "Problem Formulation",
        "bullets": [
            "กำหนดปัญหาเป็น Markov Decision Process",
            "state/observation: PPO ใช้ตำแหน่งและความเร็วโดยตรง ส่วน DDPG ใช้ error, integral error และ derivative error",
            "action: เวกเตอร์ต่อเนื่อง [Kp, Ki, Kd]",
            "reward: ให้รางวัลเมื่ออยู่ใกล้ศูนย์กลาง ลดความเร็ว และลงโทษเมื่อ fail",
            "episode จบเมื่อลูกบอลออกนอกพื้นที่ที่กำหนดหรือหล่นต่ำกว่าแพลตฟอร์ม",
        ],
    },
    {
        "title": "PPO Implementation",
        "bullets": [
            "Behavior name: pid_tuning_agent",
            "Observation size: 13 values",
            "Action size: 3 continuous values: Kp, Ki, Kd",
            "Network: 2 hidden layers, 128 hidden units",
            "Training: max_steps = 3,500,000, batch_size = 1024, buffer_size = 3000, learning_rate = 0.003",
            "Final checkpoint: pid_tuning_agent-3500018.onnx",
        ],
    },
    {
        "title": "DDPG-FC-350-E-PID Implementation",
        "bullets": [
            "Behavior name: ddpg_fc_350_e_pid_agent",
            "Observation size: 10 E-PID values ได้แก่ error, integral error, derivative error, current PID gains และ error magnitude",
            "DDPG ไม่ใส่ raw position/velocity โดยตรง แต่ error มาจากตำแหน่งลูกบอลเทียบ target และ derivative error คืออัตราการเปลี่ยน error",
            "Action size: 3 continuous values: Kp, Ki, Kd",
            "Actor/Critic network: 2 hidden layers, 350 hidden units, ELU activation",
            "Replay buffer size: 200,000; batch_size = 256",
            "Exploration noise ลดจาก 0.12 ไปถึง 0.01",
        ],
    },
    {
        "title": "Simulation Setup",
        "bullets": [
            "Unity environment ใช้ rigid-body physics, collision, contact interaction และ friction",
            "ทั้งสอง final runs ใช้ environment แบบ parallel จำนวน 5 instances",
            "PPO configuration: time_scale = 20",
            "DDPG configuration: time_scale = 50 และ no_graphics = true",
            "Fail distance = 0.55; failure penalty = -10",
        ],
    },
    {
        "title": "Result Summary",
        "bullets": [
            "PPO final cumulative reward: 179.71",
            "PPO final checkpoint reward ใน training_status.json: 179.72",
            "DDPG recent mean reward: 170.19",
            "DDPG latest episode reward: 177.14",
            "Final episode length: PPO 1,999 steps; DDPG 2,000 steps",
            "Final failure penalty: 0 สำหรับทั้งสอง algorithms",
        ],
    },
    {
        "title": "Episode Length Comparison",
        "bullets": [
            "PPO มี Environment/Episode Length ปลายทาง 1,999 steps",
            "DDPG มี Environment/Episode Length ปลายทาง 2,000 steps",
            "DDPG log มี episode length ทั้งหมด 2,122 episodes",
            "ช่วงท้ายทั้งสองวิธีอยู่ได้เกือบเต็ม episode แสดงว่าไม่ fail ในจุดสุดท้ายที่บันทึก",
        ],
        "image": "episode_length_comparison.png",
    },
    {
        "title": "Reward Comparison",
        "bullets": [
            "PPO reward เพิ่มเร็วกว่าและ converge เรียบกว่า",
            "DDPG เริ่มจาก reward ต่ำหรือติดลบ เพราะช่วง exploration แรกทำให้ fail บ่อย",
            "DDPG ดีขึ้นชัดเจนในช่วงท้าย แต่ recent mean reward ยังต่ำกว่า PPO",
        ],
        "image": "reward_comparison.png",
    },
    {
        "title": "Episode Length and Reward Breakdown",
        "bullets": [
            "ทั้งสอง policies อยู่ได้เกือบเต็ม episode ในช่วงท้าย",
            "PPO มี centered reward สูงกว่า แปลว่าลูกบอลอยู่ใกล้ศูนย์กลางกว่าโดยเฉลี่ย",
            "Low-velocity reward ของทั้งสองวิธีใกล้เคียงกัน",
            "ไม่มี failure penalty ในจุดสุดท้ายที่ log ไว้",
        ],
        "image": "reward_breakdown_final.png",
    },
    {
        "title": "Analysis",
        "bullets": [
            "PPO เสถียรกว่าใน Unity environment ปัจจุบัน เพราะ clipped updates จำกัดการเปลี่ยน policy ที่รุนแรงเกินไป",
            "DDPG เรียนรู้งานได้ แต่ไวต่อ exploration noise และ critic stability มากกว่า",
            "DDPG architecture ทำตามแนวคิด FC-350-E-PID จากงานอ้างอิง และเหมาะกับ continuous PID gain tuning",
            "ผลสุดท้ายชี้ว่า PPO เป็นตัวเลือกที่ปลอดภัยกว่าในตอนนี้ ส่วน DDPG ยังต้อง tuning เพิ่มก่อนสรุปว่าดีกว่า",
        ],
    },
    {
        "title": "Limitations and Future Work",
        "bullets": [
            "training logs ยังไม่มี evaluation trajectory แยกสำหรับ RMS position error หรือ overshoot",
            "PPO และ DDPG ใช้รายละเอียด observation/reward ต่างกันบางส่วน ทำให้การเปรียบเทียบไม่ใช่ algorithm-only แบบสมบูรณ์",
            "ควรทดสอบ ONNX สุดท้ายของทั้งสองวิธีด้วย evaluation seeds ชุดเดียวกัน",
            "ควรเพิ่มกราฟ x-y trajectory, radial error over time และ PID gains over time",
            "requirement ของรายงานระบุว่าฉบับสุดท้ายควรเขียนเป็น LaTeX และไม่เกิน 12 หน้า",
        ],
    },
    {
        "title": "Conclusion",
        "bullets": [
            "โครงงานสามารถ implement การปรับค่า PID gains ด้วย DRL ใน Unity ได้สำเร็จ",
            "PPO final run pid_tuning_multi_12 ให้ผลดีที่สุดและเสถียรที่สุด",
            "DDPG final run DDPG_14 สามารถเรียนรู้พฤติกรรมการทรงตัวได้ แต่ reward variance สูงกว่า",
            "ทั้งสอง algorithms สามารถรักษาลูกบอลให้อยู่บนแพลตฟอร์มได้ใน final logged episodes",
        ],
    },
]


def xml_text(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def paragraph(text: str, level: int = 0, size: int = 2400, bold: bool = False) -> str:
    mar_l = level * 342900
    indent = -171450 if level else 0
    bullet = '<a:buChar char="•"/>' if level else "<a:buNone/>"
    bold_attr = ' b="1"' if bold else ""
    return (
        f'<a:p><a:pPr marL="{mar_l}" indent="{indent}">{bullet}</a:pPr>'
        f'<a:r><a:rPr lang="th-TH" sz="{size}" dirty="0"{bold_attr}/>'
        f"<a:t>{xml_text(text)}</a:t></a:r><a:endParaRPr lang=\"th-TH\" sz=\"{size}\"/></a:p>"
    )


def text_box(shape_id: int, name: str, x: int, y: int, w: int, h: int, paras: list[str]) -> str:
    body = "".join(paras)
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="{xml_text(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square" rtlCol="0"/><a:lstStyle/>{body}</p:txBody>
    </p:sp>"""


def picture(shape_id: int, rel_id: str, img_name: str, x: int, y: int, w: int, h: int) -> str:
    return f"""
    <p:pic>
      <p:nvPicPr><p:cNvPr id="{shape_id}" name="{xml_text(img_name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>"""


def slide_xml(slide: dict, index: int) -> tuple[str, str | None]:
    title_y = int(0.35 * EMU_PER_INCH)
    title_h = int(0.55 * EMU_PER_INCH)
    margin_x = int(0.55 * EMU_PER_INCH)
    title = text_box(
        2,
        "Title",
        margin_x,
        title_y,
        int(SLIDE_W - 2 * margin_x),
        title_h,
        [paragraph(slide["title"], size=2600, bold=True)],
    )
    has_image = "image" in slide
    image_xml = ""
    body_w = int(SLIDE_W - 1.1 * EMU_PER_INCH)
    body_h = int(SLIDE_H - 1.35 * EMU_PER_INCH)
    if has_image:
        body_w = int(6.15 * EMU_PER_INCH)
        body_h = int(5.9 * EMU_PER_INCH)
        image_path = REPORT_ASSETS / slide["image"]
        with Image.open(image_path) as im:
            aspect = im.width / im.height
        img_w = int(5.95 * EMU_PER_INCH)
        img_h = int(img_w / aspect)
        if img_h > int(4.2 * EMU_PER_INCH):
            img_h = int(4.2 * EMU_PER_INCH)
            img_w = int(img_h * aspect)
        image_xml = picture(
            4,
            "rId2",
            slide["image"],
            int(6.9 * EMU_PER_INCH),
            int(1.45 * EMU_PER_INCH),
            img_w,
            img_h,
        )
    body_paras = [paragraph(item, level=1, size=1700 if has_image else 1900) for item in slide["bullets"]]
    body = text_box(3, "Content", margin_x, int(1.1 * EMU_PER_INCH), body_w, body_h, body_paras)
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {title}
      {body}
      {image_xml}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""
    rels = None
    if has_image:
        rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{slide['image']}"/>
</Relationships>"""
    return xml, rels


def content_types() -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slides) + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>"""


def presentation_xml() -> str:
    sld_ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>'
        for i in range(1, len(slides) + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{int(SLIDE_W)}" cy="{int(SLIDE_H)}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="th-TH"/></a:defPPr></p:defaultTextStyle>
</p:presentation>"""


def presentation_rels() -> str:
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(1, len(slides) + 1):
        rels.append(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {' '.join(rels)}
</Relationships>"""


ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

APP_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{len(slides)}</Slides>
</Properties>"""

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
CORE_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>RL Final Presentation Content Only</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""

THEME_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ContentOnly">
  <a:themeElements>
    <a:clrScheme name="Office"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F1F1F"/></a:dk2><a:lt2><a:srgbClr val="EEEEEE"/></a:lt2><a:accent1><a:srgbClr val="000000"/></a:accent1><a:accent2><a:srgbClr val="000000"/></a:accent2><a:accent3><a:srgbClr val="000000"/></a:accent3><a:accent4><a:srgbClr val="000000"/></a:accent4><a:accent5><a:srgbClr val="000000"/></a:accent5><a:accent6><a:srgbClr val="000000"/></a:accent6><a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Tahoma"/><a:ea typeface="Tahoma"/><a:cs typeface="Tahoma"/></a:majorFont><a:minorFont><a:latin typeface="Tahoma"/><a:ea typeface="Tahoma"/><a:cs typeface="Tahoma"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""

SLIDE_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

SLIDE_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

SLIDE_MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""

SLIDE_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def build() -> None:
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types())
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("docProps/app.xml", APP_XML)
        z.writestr("docProps/core.xml", CORE_XML)
        z.writestr("ppt/presentation.xml", presentation_xml())
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels())
        z.writestr("ppt/theme/theme1.xml", THEME_XML)
        z.writestr("ppt/slideMasters/slideMaster1.xml", SLIDE_MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", SLIDE_MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", SLIDE_LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", SLIDE_LAYOUT_RELS)

        copied_media = set()
        for i, slide in enumerate(slides, start=1):
            xml, rels = slide_xml(slide, i)
            z.writestr(f"ppt/slides/slide{i}.xml", xml)
            if rels:
                z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels)
                image_name = slide["image"]
                if image_name not in copied_media:
                    z.write(REPORT_ASSETS / image_name, f"ppt/media/{image_name}")
                    copied_media.add(image_name)
            else:
                z.writestr(
                    f"ppt/slides/_rels/slide{i}.xml.rels",
                    """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>""",
                )
    print(OUT)


if __name__ == "__main__":
    build()
