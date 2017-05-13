import re

def micro_tokenize(txt):
    words = []
    # split at whitespace
    for w in txt.split():
        w = w.strip('.,!?:;"-+()„“”»«…\'`’*')
        # words need to contain at least one "regular" character
        if re.search(r'[a-zöüßA-ZÄÖÜ]', w):
            words.append(w)
    return words

def normalize(txt):
    txt = txt.lower()

    # replace URLs
    url_re1 = re.compile(r'(?:ftp|http)s?://[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re1.sub('URL', txt)
    url_re2 = re.compile(r'\bwww\.[a-zA-Z0-9-]{2,63}\.[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re2.sub('URL', txt)
    url_re3 = re.compile(r'\b[a-zA-Z0-9.]+\.(?:com|org|net|io)')
    txt = url_re3.sub('URL', txt)

    # replace emoticons
    # "Western" emoticons such as =-D and (^:
    # inspired by http://sentiment.christopherpotts.net/tokenizing.html
    s = r"(^|\s)"                # beginning or whitespace required before
    s += r"(?:"                  # begin emoticon
    s += r"(?:"                  # begin "forward" emoticons like :-)
    s += r"[<>]?"                # optinal hat/brow
    s += r"[:;=8xX]"             # eyes
    s += r"[o*'^-]?"             # optional nose
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r")"                    # end "forward" emoticons
    s += r"|"                    # or
    s += r"(?:"                  # begin "backward" emoticons like (-:
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r"[o*'^-]?"             # optional nose
    s += r"[:;=8xX]"             # eyes
    s += r"[<>]?"                # optinal hat/brow
    s += r")"                    # end "backward" emoticons
    # "Eastern" emoticons like ^^ and o_O
    s += r"|"                    # or
    s += "(?:\^\^)|(?:o_O))"     # only two eastern emoticons for now
    s += r"(\s|$)"               # white space or end required after
    emoticon_re = re.compile(s)
    txt = emoticon_re.sub(r'\1EMOTICON\2', txt)

    # remove repeated symbols
    for s in ',.!?:;#-_=+*/$@%<>&()[]':
        txt = re.sub('[%s]+' % s, s, txt)

    # separate punctuation
    txt = re.sub(r'([.,!?:;/()\'"„“”»«`’…$%*])', r' \1 ', txt)

    # remove leading, trailing and repeated whitespace
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)

    return txt
