import ifcopenshell
model = ifcopenshell.open("Building-Structural.ifc")

# Print all unique types in the file
counts = {}
for product in model.by_type("IfcProduct"):
    t = product.is_a()
    counts[t] = counts.get(t, 0) + 1

print("Types found in file:")
for t, count in counts.items():
    print(f"{t}: {count}")