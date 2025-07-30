import pandas as pd


# Create the dataset as shown in the image
data = {
    "Outlook": ["Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Sunny",
                "Overcast", "Rainy", "Rainy", "Sunny", "Rainy", "Overcast",
                "Overcast", "Sunny"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
                    "Cool", "Mild", "Cool", "Mild", "Mild", "Mild",
                    "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal",
                 "Normal", "High", "Normal", "Normal", "Normal", "High",
                 "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong",
             "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
             "Weak", "Strong"],
    "Play_ball": ["No", "No", "Yes", "Yes", "Yes", "No",
                  "Yes", "No", "Yes", "Yes", "Yes", "Yes",
                  "Yes", "No"]
}

df = pd.DataFrame(data)

# Save to CSV
csv_path = r"H:/My Drive/01.UIT/HK8/03.KhaiThacDuLieuTruyenThongXaHoi/DOAN/Decision_Tree/play_ball.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")





