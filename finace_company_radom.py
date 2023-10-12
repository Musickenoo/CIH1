import random
data = []

for i in range(400):
    current_assets = round(random.uniform(100000, 1000000), 2)
    current_liabilities = round(random.uniform(50000, 500000), 2)
    total_assets = round(current_assets + random.uniform(10000, 1000000), 2)
    total_sales = round(random.uniform(50000, 500000), 2)
    non_current_liabilities = round(random.uniform(30000, 300000), 2)
    total_liabilities = round(current_liabilities + non_current_liabilities, 2)
    total_equity = round(total_assets - total_liabilities, 2)
    ebit = round(random.uniform(10000, 100000), 2)
    cash_flow_operations = round(random.uniform(20000, 200000), 2)
    net_profit = round(random.uniform(10000, 100000), 2)
    shareholders_equity = round(random.uniform(50000, 500000), 2)
    corporate_governance = random.randint(29000, 100000)

    # คำนวณค่าตามเกณฑ์ที่กำหนด
    CACL = current_assets / current_liabilities
    WCTA = (current_assets - current_liabilities) / total_assets
    CATA = current_assets / total_assets
    CFOCL = cash_flow_operations / current_liabilities
    TITA = total_sales / total_assets
    NCLTA = non_current_liabilities / total_assets
    NCLTL = non_current_liabilities / total_liabilities
    TLTA = total_liabilities / total_assets
    TLTE = total_liabilities / total_equity
    TETA = total_equity / total_assets
    EBTL = ebit / total_liabilities
    CFOTL = cash_flow_operations / total_liabilities
    ROA = net_profit / total_assets
    ROE = net_profit / shareholders_equity
    CGR = corporate_governance / 1000
    
    # เกณท์วิเคราะคะแนนความสุ่มเสี่ยง
    CACL_points = 1 if CACL < 1 else 0
    WCTA_points = 1 if WCTA < 0 else 0
    CATA_points = 1 if CATA < 0.5 else 0
    CFOCL_points = 1 if CFOCL < 0.5 else 0
    TITA_points = 1 if TITA < 0.2 else 0
    NCLTA_points = 1 if NCLTA > 0.5 else 0
    NCLTL_points = 1 if NCLTL > 0.5 else 0
    TLTA_points = 1 if TLTA > 1 else 0
    TLTE_points = 1 if TLTE > 1 else 0
    TETA_points = 1 if TETA < 0.4 else 0
    EBTL_points = 1 if EBTL < 0.4 else 0
    CFOTL_points = 1 if CFOTL < 0.5 else 0
    ROA_points = 1 if ROA < 0.3 else 0
    ROE_points = 1 if ROE < 0.3 else 0
    CGR_points = 1 if CGR < 60 else 0
    total = (CACL_points + WCTA_points + CATA_points + CFOCL_points 
             + TITA_points + NCLTA_points + NCLTL_points + TLTA_points 
             + TLTE_points + TETA_points + EBTL_points + CFOTL_points + ROA_points + ROE_points + CGR_points)
    # กำหนดค่าในแถว 13-14 ตามค่า total
    if total >= 9:
        col_13_14 = [1, 0]
    else:
        col_13_14 = [0, 1]

    data.append([ current_assets, current_liabilities, total_assets, total_sales, non_current_liabilities,
                 total_liabilities, total_equity, ebit, cash_flow_operations, net_profit, shareholders_equity,
                 corporate_governance, col_13_14[0], col_13_14[1]])

# สร้างและเขียนข้อมูลลงในไฟล์ txt
with open("financial_data.txt", "w") as file:


    # เขียนข้อมูลบริษัทและค่าการเงินและ Corporate Governance
    for i in range(400):
        data_row = "\t".join(map(str, data[i]))
        data_row += "\n"
        file.write(data_row)

print("ไฟล์ financial_data.txt ถูกสร้างเรียบร้อยและเขียนข้อมูลลงไปแล้ว")
# นับจำนวนแถวที่มีค่า 0 1
count_0_1 = sum(1 for row in data if row[12] == 0 and row[13] == 1)

# นับจำนวนแถวที่มีค่า 1 0
count_1_0 = sum(1 for row in data if row[12] == 1 and row[13] == 0)

print(f'จำนวนค่า 0 1 หรือ บริษัทที่ไม่เข้าขายอาจถูกเพิกถอน: {count_0_1} บริษัท')
print(f'จำนวนค่า 1 0 หรือ บริษัทที่เข้าขายอาจถูกเพิกถอน: {count_1_0} บริษัท')




