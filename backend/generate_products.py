import pandas as pd
import random

electronics = [
    "Apple iPhone 14",
    "Apple iPhone 14 Pro",
    "Samsung Galaxy S21",
    "Samzung Galaxy S21",  # typo
    "Sony Wireless Headphones",
    "Dell Inspiron Laptop",
    "HP Pavilion Laptop",
    "Apple MacBook Air",
]

fashion = [
    "Nike Running Shoes",
    "Adidas Sports Jacket",
    "Puma Casual T-Shirt",
    "Levis Denim Jeans",
    "Gucci Leather Belt",
    "Zara Summer Dress",
]

groceries = [
    "Premium Basmati Rice 5kg",
    "Organic Almond Milk",
    "Whole Wheat Bread",
    "Fresh Farm Eggs",
    "Olive Oil Extra Virgin",
    "Dark Chocolate Bar",
]

categories = electronics + fashion + groceries

products = []

for i in range(1, 501):
    base = random.choice(categories)

    if random.random() < 0.2:
        base += f" {random.randint(1, 3)}"

    products.append([i, base])

df = pd.DataFrame(products, columns=["product_id", "product_name"])

print("Preview:")
print(df.head())

df.to_csv("data/products.csv", index=False)

print("\n✅ CSV generated successfully!")
print("Rows:", len(df))
