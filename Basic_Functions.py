# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 11:18:42 2025

@author: Ranjan Segu
"""

####BASIC_FUNCTIONS####

#First_Function

print('Hi Ranjan')


#Assigning_a_variable

x = ("Hello World")
print(x)


#Asking_For_Help

help(str)


#String_Functions

print(x*5)

print(x + " Innit")

"m" in x
"W" in x

#Calculations_with_Variables

y = (7)

print(y)

y + 2
y - 2
y * 2
y ** 2
y % 2
y / 2


#Types_&_Type_Conversion

s = ("Hi", "Ranjan")

i = (7, 14, 21, 35)

f = (3.5)

b = (True)

b1 = (False)

b2 = (True, False)

#String_Operations

s[1]

i[0:5]

x.upper()

x.lower()

x.count("H")

x.replace("World", "Ranjan")

st= "-----Hello------"

print(st.strip("-"))


#Lists

L1 = "is"

L2 = "nice"

L3 = ["my", "list", L1, L2]

L4 = [[4, 5, 6, 7  ], [3, 4, 5, 6]]


L3[1]
L3[-3]
L3[1:3]
L3[1:]
L3[:3]
L3[:]

L3[1][1]
L3[1][:2]

L3 + L4

L4 * 2

any(x > 4 for sublist in L4 for x in sublist)


L3.index(L1)
L3.count(L1)
L3.append("!")
L3.remove("!")
del (L3[0:1])
L3.reverse()
L3.extend("!")
L3.pop(1)
L3.insert(1,"!")
L3.sort()
