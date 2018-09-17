import run
frog = run.run()

for i in range(2):
    frog.value_iteration()

print(frog.values)
print(frog.rules)