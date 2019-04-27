#x=-7
#x= -0.2
x=8

delta_t = 0.01

for i in range(1000):
    x = x + delta_t*(x-(x*x*x))
    print(x)
    
# x=-7 stabiler Fixpunkt bei -1, Minimum der integrierten Funktion (Potentialfunktion)
# x=-0.2 stabiler Fixpunkt bei -1, s.o.
# x=8 stabiler Fixpunkt bei 1, s.o.
# verläuft in Attraktor bei 1 oder -1