import sqlite3

conn = sqlite3.connect('titanic.db')
cursor = conn.cursor()

print("=== 数据库表列表 ===")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"数据库中的表: {tables}")

print("\n=== passengers 表结构 ===")
cursor.execute("PRAGMA table_info(passengers)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]} ({col[2]})")

print("\n=== 表数据 ===")
cursor.execute("SELECT * FROM passengers")
rows = cursor.fetchall()
if len(rows) == 0:
    print("  表中暂无数据")
else:
    print(f"  共 {len(rows)} 条记录")
    for row in rows[:5]:
        print(f"  {row}")

conn.close()
print("\n数据库连接测试完成!")