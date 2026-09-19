"""
Arabic -> Latin transliteration for the card's free-text fields.

IMPORTANT, AND THE REASON EVERY FIELD THIS PRODUCES IS LABELLED AS
GENERATED: Arabic-to-Latin name spelling is genuinely ambiguous. "محمد"
appears on real Egyptian passports as Mohamed, Mohammed, Muhammad and
Mohamad - all correct, none derivable from the Arabic alone, because the
official spelling is whatever that person's documents already say. This
module produces a consistent, readable rendering; it does NOT produce the
legal spelling on someone's passport, and must never be presented as if
it did.

Approach is dictionary-first, rules-second:

* Common Egyptian given names and surnames are looked up directly, so the
  frequent cases come out in their conventional spelling ("حسن" ->
  "Hassan", not the letter-by-letter "Hsn").
* Anything else falls back to a deterministic letter mapping, which gives
  a readable approximation for addresses, professions and rarer names.

No new dependency: the mapping is small and explicit, which also makes
the output predictable and reviewable - preferable here to pulling in a
general Arabic NLP toolkit for one function.
"""
import re

# Tashkeel (diacritics) and the tatweel elongation character carry no
# information for transliteration and appear inconsistently in OCR output.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٟـۖ-ۭ]")

# Conventional spellings for names common on Egyptian cards. These are the
# spellings people actually use, which letter-by-letter rules cannot
# reproduce (Arabic omits short vowels, so "حسن" has no 'a' to map).
COMMON_NAMES = {
    "محمد": "Mohamed", "أحمد": "Ahmed", "احمد": "Ahmed", "محمود": "Mahmoud",
    "مصطفى": "Mostafa", "إبراهيم": "Ibrahim", "ابراهيم": "Ibrahim",
    "حسن": "Hassan", "حسين": "Hussein", "علي": "Ali", "عمر": "Omar",
    "خالد": "Khaled", "كريم": "Karim", "طارق": "Tarek", "وليد": "Walid",
    "سامي": "Sami", "رامي": "Rami", "عادل": "Adel", "فؤاد": "Fouad",
    "ياسين": "Yassin", "محسن": "Mohsen", "سيد": "Sayed", "السيد": "El Sayed",
    "عبد": "Abd", "الله": "Allah", "عبدالله": "Abdullah",
    "العزيز": "El Aziz", "عبدالعزيز": "Abdelaziz", "الرحمن": "El Rahman",
    "عبدالرحمن": "Abdelrahman", "مينا": "Mina", "بيتر": "Peter",
    "جرجس": "Guirguis", "مرقس": "Morcos", "يوسف": "Youssef",
    "فاطمة": "Fatma", "عائشة": "Aisha", "مريم": "Mariam", "سارة": "Sara",
    "نور": "Nour", "هدى": "Hoda", "أمل": "Amal", "منى": "Mona",
    # Surnames
    "حمزة": "Hamza", "إسماعيل": "Ismail", "اسماعيل": "Ismail", "زكي": "Zaki",
    "كامل": "Kamel", "حسني": "Hosny", "سليم": "Selim", "رزق": "Rezk",
    "منصور": "Mansour", "جاد": "Gad", "بكر": "Bakr", "الشريف": "El Sherif",
    "النجار": "El Naggar", "درويش": "Darwish", "فهمي": "Fahmy",
    "عثمان": "Osman", "شعبان": "Shaaban",
    # Governorates and common address words
    "القاهرة": "Cairo", "الجيزة": "Giza", "الإسكندرية": "Alexandria",
    "الاسكندرية": "Alexandria", "الدقهلية": "Dakahlia", "المنيا": "Minya",
    "أسيوط": "Assiut", "اسيوط": "Assiut", "الشرقية": "Sharqia",
    "الغربية": "Gharbia", "البحيرة": "Beheira", "المنوفية": "Monufia",
    "بني": "Beni", "سويف": "Suef", "الفيوم": "Fayoum",
    "شارع": "Street", "ميدان": "Square", "حي": "District",
    # Residential area names that appear in card addresses - without these
    # the letter-by-letter fallback renders them vowel-sparse.
    "البنفسج": "El Banafseg", "الياسمين": "El Yasmin", "النرجس": "El Nargis",
    "الفردوس": "El Ferdous", "الأندلس": "El Andalus", "الاندلس": "El Andalus",
    "الزهراء": "El Zahraa", "الشروق": "El Shorouk", "عين": "Ain",
    "شمس": "Shams", "حدائق": "Hadayek", "القبة": "El Qobba",
    "عمارات": "Buildings", "فيلات": "Villas", "أبراج": "Towers",
    "مدينة": "City", "نصر": "Nasr", "المعادي": "Maadi", "الزمالك": "Zamalek",
    "المهندسين": "Mohandessin", "التجمع": "Settlement", "الاول": "First",
    "الأول": "First", "الثالث": "Third",
    # Professions
    "مهندس": "Engineer", "طبيب": "Doctor", "محاسب": "Accountant",
    "مدرس": "Teacher", "موظف": "Employee", "طالب": "Student",
    "محامي": "Lawyer", "صيدلي": "Pharmacist", "تاجر": "Merchant",
    "سائق": "Driver", "بكالوريوس": "Bachelor", "دبلوم": "Diploma",
    "علوم": "Science", "تجارة": "Commerce", "هندسة": "Engineering",
    "الحاسب": "Computer", "فني": "Technical", "ربة": "Housewife",
    "منزل": "Home", "فى": "of", "في": "of",
}

# Letter-by-letter fallback. Egyptian convention where it differs from
# Modern Standard Arabic (notably ج -> "g", as in "Gamal" rather than
# "Jamal"), since these are Egyptian cards.
_LETTER_MAP = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ب": "b", "ت": "t", "ث": "th",
    "ج": "g", "ح": "h", "خ": "kh", "د": "d", "ذ": "z", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "a",
    "غ": "gh", "ف": "f", "ق": "k", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "ة": "a", "و": "w", "ي": "y", "ى": "a", "ئ": "e", "ؤ": "o",
    "ء": "", "لا": "la",
}

_ARABIC_INDIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _transliterate_word(word: str) -> str:
    """Dictionary lookup first, then letter-by-letter."""
    if word in COMMON_NAMES:
        return COMMON_NAMES[word]

    # The definite article "ال" is written joined to its noun. Strip it,
    # render the bare noun, and re-attach "El" - otherwise "البنفسج"
    # letter-maps to the unreadable "Albnfsg" instead of "El Banafseg".
    if len(word) > 2 and word.startswith("ال"):
        stem = word[2:]
        if stem in COMMON_NAMES:
            return f"El {COMMON_NAMES[stem]}"
        return f"El {_letter_map_word(stem)}"

    return _letter_map_word(word)


def _letter_map_word(word: str) -> str:
    """Last-resort letter-by-letter rendering. Necessarily imperfect:
    Arabic does not write short vowels, so an unknown word comes out
    vowel-sparse ("بنفسج" -> "Bnfsg"). That is why COMMON_NAMES carries
    the words that actually recur on these cards."""
    out = "".join(_LETTER_MAP.get(ch, ch) for ch in word)
    return out.capitalize() if out else out


def transliterate(text: str) -> str:
    """
    Renders Arabic text in Latin script.

    Returns "" for empty input, and passes through anything already in
    Latin script (a card serial like "AB1234567") unchanged. Arabic-Indic
    digits are converted to Western ones so an address's house number is
    readable.

    See this module's docstring: the result is a generated rendering, not
    an official or legal spelling.
    """
    if not text:
        return ""

    text = _ARABIC_DIACRITICS.sub("", text).translate(_ARABIC_INDIC_TO_WESTERN)

    words = []
    for word in text.split():
        # Keep punctuation-only and already-Latin tokens as they are.
        if not any("ء" <= ch <= "ي" for ch in word):
            words.append(word)
            continue
        words.append(_transliterate_word(word))

    return " ".join(w for w in words if w).strip()
