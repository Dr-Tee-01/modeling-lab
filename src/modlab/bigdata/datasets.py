"""The datasets used on the IE494 Big Data Processing slides, reproduced exactly.

Where a slide is internally inconsistent, the version used here is the consistent one and the
choice is noted. Every lab view imports its data from here, so the numbers on screen are the
ones from the lectures.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- MapReduce
# L3 slide 17 / MR deck p4-6: 30 employee rows over 3 data nodes, map output (dno, salary).
# (L3 slide 16 and MR p3 show different dno values for some people; slides 17 / p4-6 are the
#  internally consistent version, so they are used.)
EMP30_NODES: list[list[tuple[int, int]]] = [
    [(5, 62506), (3, 104437), (5, 64955), (5, 64991), (5, 50825), (5, 57568), (4, 95660), (5, 59365), (5, 47837), (3, 50178)],
    [(5, 54670), (5, 47211), (3, 92328), (5, 58709), (5, 52505), (5, 57834), (5, 70131), (5, 59026), (3, 110000), (5, 53250)],
    [(5, 51044), (5, 64919), (5, 62910), (5, 66441), (5, 57815), (3, 103613), (1, 106367), (5, 74312), (5, 53492), (1, 63000)],
]

# MR deck p18 / p30 (IBM figure): WordCount over three input splits, range-partitioned A-I, J-Q, R-Z.
WORDCOUNT_SPLITS = ["Hadoop Map Map Reduce Hadoop Map Map", "Map Map Reduce Map Hadoop", "Key Key Value Value"]
WORDCOUNT_RANGES = [("A", "I"), ("J", "Q"), ("R", "Z")]

# MR deck 2 p24: Top-3 customers by amount, two input splits.
TOPN_SPLITS = [[("C1", 3500), ("C2", 19000), ("C3", 31000), ("C4", 5000)],
               [("C5", 5500), ("C6", 40000), ("C7", 2300), ("C8", 21000), ("C9", 7600)]]

# MR deck 2 p12/p17: repartition-join example.
MR_CUSTOMERS = [(111, "India", "Gujarat"), (112, "USA", "Texas"), (114, "UK", "Scotland")]
MR_ORDERS = [(1, 111, 550), (2, 112, 780), (6, 111, 980), (9, 114, 855), (11, 111, 899), (12, 112, 799),
             (14, 114, 718), (17, 112, 324), (19, 114, 440)]

# --------------------------------------------------------------------------- Spark RDDs
# RDD deck 1 p28: 8 employees (eno, name, dno, salary). (p27-28 index the wrong fields; p34 is right.)
EMP8 = [(101, "John", 5, 30000), (102, "Franklin", 5, 40000), (103, "Joyce", 5, 25000), (104, "Ramesh", 5, 38000),
        (105, "James", 1, 55000), (106, "Jennifer", 4, 43000), (107, "Ahmad", 4, 25000), (108, "Alicia", 4, 25000)]

# RDD deck 1 p40-41: customers and 20 orders cycling through customers 111-115.
SPARK_CUSTOMERS = [(111, "Gujarat"), (112, "Texas"), (113, "Maharashtra"), (114, "Scotland"), (115, "Washington"),
                   (116, "Karnataka"), (117, "Gujarat")]
_AMOUNTS = [550, 780, 830, 320, 230, 980, 320, 325, 855, 355, 899, 799, 620, 718, 634, 873, 324, 413, 440, 845]
SPARK_ORDERS = [(i + 1, 111 + i % 5, a) for i, a in enumerate(_AMOUNTS)]

# Spark implementation deck p43/45: 12 customers in 3 partitions of 4.
REGIONS12 = [(1, "Adam Hernandez", "Northeast"), (2, "Keith Griffin", "Northeast"), (3, "Jerry Green", "West"),
             (4, "Chris Armstrong", "Northeast"), (5, "Stephen Payne", "South"), (6, "Joshua Bennett", "Northeast"),
             (7, "Shawn Cook", "Midwest"), (8, "George Lewis", "West"), (9, "Joshua Ryan", "Midwest"),
             (10, "Jonathan Hawkins", "West"), (11, "Joshua Little", "South"), (12, "Carl Nguyen", "Midwest")]

# Spark SQL DDL/DML deck p39-40: 19 marks and the letterGrade UDF thresholds (the slide repeats "DD"
# for >35; DE is used for (30, 35] as in its own worked output, and >35 keeps DD as written).
MARKS19 = [44, 30, 36, 34, 68, 38, 60, 68, 68, 44, 82, 48, 42, 60, 54, 58, 42, 68, 66]

# --------------------------------------------------------------------------- Data warehousing
# DW deck slide 18 (Han & Kamber): sales by city x quarter x item type.
HK_ITEMS = ["home entertainment", "computer", "phone", "security"]
HK_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
HK_SALES = {
    "Chicago":   [[854, 882, 89, 623], [943, 890, 64, 698], [1032, 924, 59, 789], [1129, 992, 63, 870]],
    "New York":  [[1087, 968, 38, 872], [1130, 1024, 41, 925], [1034, 1048, 45, 1002], [1142, 1091, 54, 984]],
    "Toronto":   [[818, 746, 43, 591], [894, 769, 52, 682], [940, 795, 58, 728], [978, 864, 59, 784]],
    "Vancouver": [[605, 825, 14, 400], [680, 952, 31, 512], [812, 1023, 30, 501], [927, 1038, 38, 580]],
}
# Lattice deck slide 10: location hierarchy (city -> province/state -> country).
LOCATION = {"Vancouver": ("British Columbia", "Canada"), "Victoria": ("British Columbia", "Canada"),
            "Toronto": ("Ontario", "Canada"), "Ottawa": ("Ontario", "Canada"),
            "New York": ("New York", "USA"), "Buffalo": ("New York", "USA"),
            "Chicago": ("Illinois", "USA"), "Urbana": ("Illinois", "USA")}

# DW deck slide 24: 9-row fact table (store, item, customer, price); total 735.
FACT9 = [("s1", "i1", "c1", 10), ("s1", "i1", "c2", 15), ("s1", "i2", "c1", 30), ("s1", "i2", "c2", 80),
         ("s1", "i3", "c1", 105), ("s2", "i1", "c2", 65), ("s2", "i2", "c1", 280), ("s2", "i2", "c2", 90),
         ("s2", "i3", "c2", 60)]

# DW deck slide 29: 30-row fact table (item, store, customer, amount); total 3350.
FACT30 = [("item1", "s1", "c1", 10), ("item1", "s1", "c2", 15), ("item1", "s1", "c3", 45), ("item1", "s2", "c2", 65),
          ("item2", "s1", "c1", 30), ("item2", "s1", "c2", 80), ("item2", "s1", "c3", 40), ("item2", "s2", "c1", 280),
          ("item2", "s2", "c2", 90), ("item2", "s2", "c3", 145), ("item2", "s2", "c4", 165), ("item2", "s3", "c2", 165),
          ("item2", "s3", "c3", 145), ("item2", "s6", "c3", 95), ("item2", "s6", "c4", 90),
          ("item3", "s1", "c1", 105), ("item3", "s2", "c2", 60), ("item3", "s3", "c2", 165), ("item3", "s3", "c3", 150),
          ("item3", "s4", "c1", 65), ("item3", "s4", "c2", 70), ("item3", "s6", "c4", 165),
          ("item4", "s4", "c3", 90), ("item4", "s5", "c1", 180), ("item4", "s5", "c2", 70), ("item4", "s5", "c4", 170),
          ("item4", "s6", "c4", 145), ("item5", "s5", "c2", 155), ("item5", "s6", "c3", 175), ("item5", "s6", "c4", 125)]

# DW deck slide 30 (Gupta): enrolments by country x degree, all semesters.
DEGREES = ["BSc", "LLB", "MBBS", "BCom", "BIT"]
ENROL = {"Australia": [12, 30, 31, 103, 21], "India": [19, 0, 32, 47, 30], "Malaysia": [10, 2, 29, 31, 43],
         "Singapore": [4, 4, 20, 22, 54], "Sweden": [13, 0, 10, 41, 14], "UK": [9, 28, 40, 46, 24],
         "USA": [4, 4, 30, 25, 31]}

# DW deck slide 32: monthly sales (Mar-Nov) for a pivot / windowing example.
PIVOT_MONTHS = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
PIVOT_SALES = {"CSCL": [870, 3884, 9000, 9792, 11691, 14625, 15068, 13230, 7290],
               "DNA Ligase": [420, 1650, 3360, 3364, 3960, 4860, 5348, 6600, 3888],
               "Phenol": [270, 1096, 2245, 2361, 2376, 3210, 3810, 4410, 2430],
               "Pipette Tip": [196, 1550, 3800, 4560, 6432, 6888, 6440, 3520, 1952]}

# --------------------------------------------------------------------------- HDFS (L3 slide 7)
HDFS_BLOCK_MB = 128
HDFS_REPLICATION = 3
HDFS_LAYOUT = {"DN1": ["b1", "b4", "b7", "b2", "b5"], "DN2": ["b1", "b5", "b8", "b2", "b6"],
               "DN3": ["b3", "b6", "b9", "b4", "b7"], "DN-N": ["b3", "b8", "b9", "b5"]}
HDFS_FILES = {"file1": ["b1", "b2", "b3"], "file2": ["b4", "b5", "b6"], "file3": ["b7", "b8", "b9"]}
