"""wisdow
==================

as wise as one can be

"""

w = "ᄐᆊᄟᄍᅻ‘ᄂᄧᅍᅈჳᅐᅇᄴᅑᄃᅄᅓᅃᅁᅙᅘᅀჾᅃᄷᅛᅄᅁᅒᅃᄺᅊᄃᅊᄿᅕჳᅙᅋᄸჾᅈᅅᅎᅈᅁᅂᅕჳᅜᅈჳᅋᅃᄷᅊᄃᄴᅊᅑᅁᅌᄃᅇᅆᅇჳᅜᅄᅌჾ”ᅻᄏᄝᆊᄌ"
i = 4242
s = [80, 65, 83, 81, 65, 76]
e = "".join([chr(o) for o in [ord(l) - i - s[_ % len(s)] for _, l in enumerate(w)]])

print(e)
