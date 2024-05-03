import html
import re


def prep_text(x):
    x = re.sub(r"\s+", html.unescape(x), " ").strip()[:30700]
    return "nimic" if x == "" else x
