"""
prompts.py — LLM-toelichting stap 3 (detailpaneel): multimodaal (PNG + data).

Eén statische opening per sessie; geen chat in de studie-UI.
"""

import os as _os

OPENAI_API_KEY = (
    _os.environ.get("OPENAI_API_KEY", "").strip()
    or _os.environ.get("OPENAI_API", "").strip()
)
OPENAI_MODEL = _os.environ.get("OPENAI_MODEL", "o4-mini-2025-04-16").strip()

# ── Systeemprompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Je bent een uitlegassistent in een JADS-onderzoek. Je helpt IC-artsen het detailpaneel \
bij een geschatte kans op intrahospitaal overlijden te lezen — altijd als modeluitvoer, \
nooit als klinische waarheid.

Taal: alleen professioneel Nederlands. Geen Engels. Geen opmaak (geen sterretjes, \
geen #-koppen) — alleen platte tekst met regelafbrekingen. In je antwoord aan de arts: \
gebruik formele taal.

Studie-interface: één vaste openingstekst onder het detailpaneel. Geen chat, geen \
uitnodiging om vragen te stellen, geen afsluitende syntheseregel over het paneel.

Kop (exact één regel, zoals in de gebruikersprompt; daarna lege regel; dan één tot \
drie alinea's):

Wat betekent dit voor deze patiënt — volgens het model?

De arts ziet het plaatje al. Herhaal geen panelen, kleuren, paarse markeringen of \
bijdrage-percentages. Je mag meetnamen noemen uit de gegeven data — zonder elke waarde \
voor te lezen.

Correlationeel: patronen uit historische IC-data (MIMIC), toegepast op deze opname. \
Geen causaliteit, geen behandeladvies. Zeg niet wat de arts moet denken of herkennen.

Kenmerkregels bevatten [referentie: …] en [binnen / onder / boven referentie]. \
Zeg «onder» of «boven» referentie alleen als die status dat zegt — verzin geen \
labafwijking.

Geen ML-jargon (KAN, spline, activatie, gradient). Geen IC-subtype (MICU, SICU, …); \
alleen IC-opname. Geen «grafiek 1/2/3»; zeg dit paneel of de figuur naast het venster.

Appendix «VOLLEDIG MODELSIGNAAL PER KENMERK» in het gebruikersbericht is alleen voor \
je controle — nooit plakken of kolomtabellen in het antwoord.
"""

# ── Gebruikersprompt detailpaneel (grafiek 3) ───────────────────────────────────
_DETAIL_PANEL_INTRO = """\
Bijgevoegd: dezelfde afbeelding als het detailpaneel op stap 3.

IC-opname. Demografie staat in het profiel; niet in de twaalf panelen.
Voorspelde kans intrahospitaal overlijden: {prob:.1f}%

Samenvatting (niet het plaatje herhalen; regels onderaan zijn intern voor je):
{clinical_summary}

Twaalf panelen — patientwaarde, modelactivatie en aandeel (niet als lijst voorlezen):
{feature_rows}

Schrijf één toelichting: negen tot twaalf zinnen totaal, drie alinea's (lege regel tussen \
alinea's). Vloeiend Nederlands. Precies één kop op een eigen regel:

Wat betekent dit voor deze patiënt — volgens het model?

Alinea 1 (twee zinnen): Noem de voorspelde kans ({prob:.1f}%) en in één zin de balans: \
welke metingen trekken het modelsignaal vooral omhoog en welke omlaag. Gebruik concrete \
namen uit de samenvatting of kenmerkregels hierboven. Geen percentages per \
kenmerk; beschrijf het plaatje niet (geen rood/blauw/paars).

Alinea 2 (vijf tot zes zinnen): Dit is een correlatieve schatting uit patronen in \
historische IC-gegevens, geen bewijs dat één meting de uitkomst veroorzaakt. De \
{prob:.1f}% is een modelschatting en geen behandeladvies. Leg kort uit dat interpretatie \
altijd in klinische context moet en dat samenhang in retrospectieve data geen causaliteit \
betekent. Sluit af met één zin over waarvoor dit soort uitleg wél van waarde kan zijn: \
het inzichtelijk en transparant maken van machine learning-modellen, zodat clinici beter \
kunnen begrijpen hoe een model redenert en wanneer ze de uitkomst kunnen vertrouwen of \
bevragen.

Alinea 3 (twee tot vier zinnen): Noem het meest opvallende patroon dat voor een clinicus niet \
vanzelfsprekend is. Kijk vooral naar: [boven referentie] \
of [onder referentie] met (lagere kans); en anders naar [binnen referentie] met (hogere kans) \
— een normale waarde die toch het modelsignaal omhoog trekt. Sla kenmerken over met [binnen referentie] \
én (lagere kans) — dat is verwacht. Benoem het voorzichtig (kan, mogelijk) en leg \
kort uit waarom dit patroon in retrospectieve data kan voorkomen. Herhaal geen \
meetnaam uit alinea 1. Als echt niets opvalt: één zin over het isolated-panel \
principe versus het volledige model.

Geen tweede kop. Geen chat-uitnodiging.
"""


def format_graph_intro(
    graph_id: int,
    prob: float,
    features: list,
    clinical_summary: str = "",
) -> str:
    """Gebruikersprompt voor de opening van het detailpaneel (alleen graph_id == 3)."""
    if graph_id != 3:
        raise ValueError(f"Alleen detailpaneel (graph_id=3); ontvangen: {graph_id}")

    rows = []
    for f in features:
        name = f["name"]
        pct = f.get("contribution_pct", 0.0)
        pval = f.get("patient_value", "")
        act = f.get("activation", 0.0)
        direction = f.get("direction", "neutraal")
        nrange = f.get("normal_range", "")
        ref_status = f.get("reference_status", "")
        sign = "+" if pct >= 0 else ""
        rows.append(
            f"  {name}: patientwaarde = {pval}"
            + (f" [referentie: {nrange}]" if nrange else "")
            + (f" [{ref_status}]" if ref_status else "")
            + f", bijdrage bij deze waarde = {act:+.3f} ({direction}), "
            f"aandeel in het risicoscore-signaal = {sign}{pct:.1f}%"
        )

    summary = (clinical_summary or "").strip() or (
        "  (Geen samenvatting — gebruik de kenmerkregels hieronder.)"
    )

    return _DETAIL_PANEL_INTRO.format(
        prob=prob,
        clinical_summary=summary,
        feature_rows="\n".join(rows),
    )


def _vision_user_content(user_text: str, image_png_base64: str) -> list:
    url = f"data:image/png;base64,{image_png_base64}"
    return [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": url}},
    ]


def build_intro_messages(system: str, user_text: str, image_png_base64: str) -> list:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _vision_user_content(user_text, image_png_base64)},
    ]
