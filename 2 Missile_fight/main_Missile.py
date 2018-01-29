import Missile

A = Missile.MissileAI(first=True)
B = Missile.MissileAI(first=False)

print(A.reset())
print(B.reset())
s = A.step([1,3])
B.share(s[0])
print(B.step([2,4]))