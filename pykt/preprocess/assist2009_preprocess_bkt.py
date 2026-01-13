# _*_ coding:utf-8 _*_
import pandas as pd
from .utils import sta_infos, write_txt, format_list2str

# Added BKT columns to KEYS (not strictly used by sta_infos for basic stats but helps tracking)
KEYS = ["user_id", "skill_id", "problem_id"]

def read_data_from_csv(read_file, write_file):
    stares = []

    # Load with augmented columns
    df = pd.read_csv(read_file, encoding = 'ISO-8859-1', low_memory=False)
    
    # Match standard PyKT sorting by adding tmp_index
    df['tmp_index'] = range(len(df))
    
    # We already filtered NAs in the augmentation script, but safety first
    _df = df.dropna(subset=["user_id","problem_id", "skill_id", "correct", "order_id", "bkt_p_correct", "bkt_mastery"])

    # Basic stats (Standard pykt keys)
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"BKT Augmented Raw Stats - interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}")

    ui_df = _df.groupby('user_id', sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        # Exact match to standard pykt sort: order_id followed by original row index
        tmp_inter = tmp_inter.sort_values(by=['order_id', 'tmp_index'])
        seq_len = len(tmp_inter)
        
        seq_problems = tmp_inter['problem_id'].tolist()
        seq_skills = tmp_inter['skill_id'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        
        # Augmented BKT columns
        seq_bkt_p = tmp_inter['bkt_p_correct'].tolist()
        seq_bkt_m = tmp_inter['bkt_mastery'].tolist()
        
        # Timestamps and usetimes (NA for assist2009)
        seq_start_time = ['NA']
        seq_response_cost = ['NA']

        # Format as 8 lines per user (Added 2 lines for BKT)
        # Standard: 1.uid,len 2.qs 3.cs 4.ans 5.ts 6.cost
        # Augmented: 7.bkt_p 8.bkt_m
        user_inters.append([
            [str(user), str(seq_len)], 
            format_list2str(seq_problems), 
            format_list2str(seq_skills), 
            format_list2str(seq_ans), 
            seq_start_time, 
            seq_response_cost,
            format_list2str(seq_bkt_p),
            format_list2str(seq_bkt_m)
        ])

    write_txt(write_file, user_inters)
    print("Preprocessing to data.txt complete (8-line format)")
    return
