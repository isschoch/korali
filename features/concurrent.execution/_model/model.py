#!/usr/bin/env python3
import math
import sys

a = 20.0
b = 0.2
c = 2. * math.pi

s1 = 0
s2 = 0
for i in range(1, len(sys.argv)):
  val = float(sys.argv[i])
  s1 += val * val
  s2 += math.cos(c * val)

result = -a * math.exp(-b * math.sqrt(s1 / 4)) - math.exp(
    s2 / 4) + a + math.exp(1.)
print(-result)
