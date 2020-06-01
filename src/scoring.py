import numpy as np
import pandas as pd
import sys
import re

# question type definition
S = 0   # [S, col, corr [,rate]]
MS = 1  # [MS, [cols,..], [corr,..] [,rate]]
Num = 2 # [Num, [cols,..], [corr,..] [,rate]]
SS = 3  # [SS, [start,end], [corr,...] [,rate]]

# the list of question type and reference
# [type, column, answer[, num_candidate]]
QuestionReferences = None

def get_num_squestions(qref):
    numq = 0
    for q in qref:
        if q[0] == MS:
            numq += len(q[1])
        elif q[0] == SS:
            numq += q[1][1]-q[1][0]+1
        else: numq += 1
    return numq

def ascoringS(answer, q):
    if answer[q[1]] == q[2]:
        return 1
    else:
        return 0

def ascoringMS(answer, columns, i, ref):
    ans = answer[columns]
    if ref[i] in ans:
        return 1
    else:
        return 0

def ascoringSS(answer, i, columns, ref):
    ans = answer[columns[0]+i]
    if ans == ref[i]:
        return 1
    else:
        return 0

def ascoringNum(answer, columns, ref):
    for i,p in enumerate(columns):
        if answer[p] != ref[i]:
            return 0
    return 1

def ascoring(df, q):
    if q[0] == S:
        return df.apply(ascoringS, axis=1, raw=True, args=(q,))
    elif q[0] == MS:
        res = None 
        for i in range(len(q[2])):
            rr = df.apply(ascoringMS, axis=1, raw=True, args=(q[1], i,q[2]))
            if res is None:
                res = rr
            else:
                res = pd.concat([res, rr], axis=1)
        return res
    elif q[0] == Num:
        return df.apply(ascoringNum, axis=1, raw=True, args=(q[1], q[2]))
    elif q[0] == SS:
        res = None
        for i in range(q[1][1]-q[1][0]+1):
            rr = df.apply(ascoringSS, axis=1, raw=True, args=(i, q[1], q[2]))
            if res is None:
                res = rr
            else:
                res = pd.concat([res, rr], axis=1)
        return res
    else:
        print(f"ERROR: Undefined question type: {q[0]}")
        exit()

def get_maxcolms(qref):
    maxcol = 0
    for q in qref:
        if q[0] == S:
            if maxcol < q[1]: maxcol = q[1]
        else:
            if maxcol < max(q[1]): maxcol = max(q[1])
    return maxcol

def get_sq2p(qref):
    num_squestions = get_num_squestions(qref)
    sq2p = np.zeros(num_squestions, dtype=np.int)
    numq = 0
    numsq = 0
    for q in qref:
        if q[0] == MS:
            for i in q[1]:
                sq2p[numsq] = numq
                numsq += 1
        elif q[0] == SS:
            for i in range(q[1][1]-q[1][0]+1):
                sq2p[numsq] = numq
                numsq += 1
        else:
            sq2p[numsq] = numq
            numsq += 1
        numq += 1
    return sq2p

def correctRate(scorelist_v):
    return sum(scorelist_v) / len(scorelist_v)

def print_crate(marubatu, points_alloc):
    print("====================================", file=sys.stderr)
    print("Correct rate for each small question", file=sys.stderr)
    print("      and allocation of points", file=sys.stderr)
    print(" No: rate, points, q_type", file=sys.stderr)
    print("------------------------------------", file=sys.stderr)
    crate = marubatu.iloc[:,1:].apply(correctRate, raw=True)
    sq2p = get_sq2p(QuestionReferences)
    for i,rate in enumerate(crate):
        q = QuestionReferences[sq2p[i]]
        if q[0] == S: kind = f'  S[{q[1]}]'
        elif q[0] == MS: kind = f' MS{q[1]}'
        elif q[0] == SS: kind = f' SS{q[1]}'
        else: kind = f'Num{q[1]}'
        print(f"{i+1:3d}:{rate*100.0:3.0f}%, {points_alloc[i]:2}, {kind:}", file=sys.stderr)

def totalscore(scorelist, points_alloc):
    if len(scorelist) != len(points_alloc)+1:
        print("ERROR: in totalscore()", file=sys.stderr)
        print(scorelist, file=sys.stderr)
        print(points_alloc, file=sys.stderr)
        exit()
    return sum(scorelist[1:] * points_alloc)
#    return sum(scorelist[1:]) * 3

def get_points_alloc(qref, desired_pscore):
    num_squestions = get_num_squestions(qref)
    points_alloc = np.zeros(num_squestions, dtype=np.int)
    num = 0
    sum_palloc = 0
    for q in qref:
        weight = 100
        if len(q) >= 4:
            weight = q[3]
        if q[0] == MS:
            inum = len(q[1])
        elif q[0] == SS:
            inum = q[1][1]-q[1][0]+1
        else:
            inum = 1
        for i in range(inum):
            points_alloc[num] = weight
            sum_palloc += weight
            num += 1
    basic_unit_float = desired_pscore * 100.0 / sum_palloc
    for i in range(num_squestions):
        points_float = desired_pscore * points_alloc[i] / sum_palloc
        points = round(points_float)
        if points <= 0: points = 1
        points_alloc[i] = points
    return points_alloc, basic_unit_float

def marksheetScoring(filename, crate, desired_pscore):
    maxcolms = get_maxcolms(QuestionReferences)
    df = pd.read_csv(filename, header=None, dtype=object, skipinitialspace=True, usecols=list(range(maxcolms+1)))
    df.fillna('-1', inplace=True)
    df.replace('*', -1, inplace=True) # multi-mark col.
    df = df.astype('int')
    df[0] = df[0]+200000000
    df = df.sort_values(by=0, ascending=True)
    print(f"Marksheet-answer: #students={df.shape[0]}, #columns={df.shape[1]}(including id-number)", file=sys.stderr)
    marubatu = df[[0]]
    for q in QuestionReferences:
        ascore = ascoring(df, q)
        marubatu = pd.concat([marubatu, ascore], axis=1, ignore_index=True)
    marubatu.to_csv(filename+'.marubatu', index=False, header=False)
    points_alloc, basic_unit_float = get_points_alloc(QuestionReferences, desired_pscore)
    perfect_score = sum(points_alloc)
    print(f"#Small_questions={len(points_alloc)}", file=sys.stderr)
    print(f"Perfect_score={perfect_score} (desired_perfect_score={desired_pscore})", file=sys.stderr)
    basic_point_unit = round(basic_unit_float)
    basic_point_unit = basic_point_unit if basic_point_unit >= 1 else 1
    print(f"Basic_points_unit(weight=100)={basic_point_unit}, (float_unit = {basic_unit_float:5.2f})", file=sys.stderr)
    if crate:
        print_crate(marubatu, points_alloc)
    id_scores = pd.concat([marubatu[0], marubatu.apply(totalscore, axis=1, raw=True, args=(points_alloc,))], axis=1, ignore_index=True)
#    scores = marubatu.apply(totalscore, axis=1, raw=True, args=(points_alloc))
#    print(scores, file=sys.stderr)
#    id_scores = pd.concat([marubatu[0], scores], axis=1, ignore_index=True)
    return id_scores

### for Twins upload file

def read_twins_upload_file(twinsfilename):
    twins = pd.read_csv(twinsfilename, skiprows=1, header=None, skipinitialspace=True)
    twins.columns=['科目番号', '学籍番号', '学期区分', '学期評価', '総合評価']
    twins['総合評価'].fillna('0', inplace=True)
#    scores = twins['総合評価'].astype(int, inplace=True) # inplaceは働かない
    scores = twins['総合評価'].astype(int)
    del twins['総合評価']
    twins = pd.concat([twins, scores], axis=1, ignore_index=True)
    twins.columns=['科目番号', '学籍番号', '学期区分', '学期評価', '総合評価']
    id_scores = pd.concat([twins['学籍番号'], twins['総合評価']], axis=1, ignore_index=True)
    return id_scores, twins


### ajusting
# adjust
def point_adjust(point, xp, yp, xmax):
    gradient1 = yp / xp
    gradient2 = (xmax - yp)/(xmax - xp)
    if point <= xp:
        point = gradient1 * point
    elif point <= xmax:
        point = gradient2 * point + (xmax * (1.0-gradient2))
    return point

def adjust(id_scores, params):
    xp, yp, xmax = params
    adjustfunc = lambda p: point_adjust(p, xp, yp, xmax)
    id_scores = pd.concat([id_scores[0], id_scores[1].map(adjustfunc).astype(int)], axis=1, ignore_index=True)
    return id_scores

# a2djust

def get_points_abcd(params, id_scores):
    score_list = np.sort(id_scores[1])[::-1]
    num = len(score_list)
    points_list = [score_list[0]]
    cp = 0
    for p in params:
        cp += p
        points_list.append(score_list[round(num * cp / 100.0)])
    return points_list

def point_a2djust(p, p_max, p_ap, p_a, p_b, p_c):
    if p >= p_ap:
        newpoint = 90 + (10/(p_max-p_ap)) * (p-p_ap)
    elif p >= p_a:
        newpoint = 80 + (10/(p_ap-p_a)) * (p-p_a)
    elif p >= p_b:
        newpoint = 70 + (10/(p_a-p_b)) * (p-p_b)
    elif p >= p_c:
        newpoint = 60 + (10/(p_b-p_c)) * (p-p_c)
    else:
        newpoint = (60.0/p_c) * p
    return round(newpoint)

def a2djust(id_scores, params):
#    rate_ap, rate_a, rate_b, rate_c = params
    p_max, p_ap, p_a, p_b, p_c = get_points_abcd(params, id_scores)
    print(f"A2djust: rate_ap={params[0]}, rate_a={params[1]}, rate_b={params[2]}, rate_c={params[3]}", file=sys.stderr)
    print(f"A2djust: p_max={p_max}, p_ap={p_ap}, p_a={p_a}, p_b={p_b}, p_c={p_c}", file=sys.stderr)
    a2djustfunc = lambda p: point_a2djust(p, p_max, p_ap, p_a, p_b, p_c)
    new_id_scores = pd.concat([id_scores[0], id_scores[1].map(a2djustfunc).astype(int)], axis=1, ignore_index=True)
    return new_id_scores


# interval
def finterval(x, minval, maxval):
    if x < minval: return minval
    elif x > maxval: return maxval
    else: return x

def interval(id_scores, minmax):
    min, max = minmax
    func = lambda x: finterval(x, min, max)
    scores = id_scores.iloc[:,1].map(func).astype(int)
    id_scores = pd.concat([id_scores[0], scores], axis=1, ignore_index=True)
    return id_scores


#### print statistics

Pgakuruimei = re.compile(r'.*学群(.+学類).*')
def ex_gakuruimei(str):
    mobj = Pgakuruimei.match(str)
    if mobj:
        return mobj.group(1)
    if str.find('体育専門学群') != -1:
        return '体育専門学群'
    if str.find('芸術専門学群') != -1:
        return '芸術専門学群'
    return '不明学類'

def read_meibo(filename):
    meibo = pd.read_csv(filename, skiprows=4, header=None, skipinitialspace=True)
    if meibo[0][0] != 1:
        print("Score Error in reading meibo file.", file=sys.stderr)
        exit()
    meibo = meibo[[3,1,2,4,5]]
    meibo.columns = ['学籍番号', '所属学類', '学年', '氏名', '氏名カナ']
    meibo['所属学類'] = meibo['所属学類'].map(ex_gakuruimei)
    return meibo

def mk_gakurui_dicset(meibo):
    dicset = {}
    for i in range(meibo.shape[0]):
#        gakuruimei = ex_gakuruimei(meibo['所属学類'][i])
        gakuruimei = meibo['所属学類'][i]
        if gakuruimei in dicset:
            dicset[gakuruimei].add(meibo['学籍番号'][i])
        else:
            dicset[gakuruimei] = set([meibo['学籍番号'][i]])
    return dicset

def gakurui_statistics(id_scores, meibofilename):
    meibo = read_meibo(meibofilename)
    gdicset = mk_gakurui_dicset(meibo)
    res = []
    for gname in gdicset:
        aset = gdicset[gname]
        selectstudents = [no in aset for no in id_scores.iloc[:,0]]
        scores = id_scores.iloc[:,1][selectstudents]
        res.append([gname, scores.describe()])
    return res

def print_stat(scores):
    print("==================", file=sys.stderr)
    print("Score statistics", file=sys.stderr)
    print("------------------", file=sys.stderr)
    print(scores.describe(), file=sys.stderr)

def print_stat_gakurui(id_scores, meibofilename):
    gakurui_sta_list = gakurui_statistics(id_scores, meibofilename)
    print("==================", file=sys.stderr)
    print("Gakurui statistics", file=sys.stderr)
    print("------------------", file=sys.stderr)
    notfirst = False
    for gakuruiinfo in gakurui_sta_list:
        if notfirst:
            print('-------', file=sys.stderr)
        else:
            notfirst = True
        print(gakuruiinfo[0], file=sys.stderr)
        print(gakuruiinfo[1], file=sys.stderr)

def print_abcd(scores):
    all = len(scores)
    aplus = scores[scores>=90]
    a = scores[scores<90]
    aa = a[a>=80]
    b = scores[scores<80]
    bb = b[b>=70]
    c = scores[scores<70]
    cc = c[c>=60]
    d = scores[scores<60]
    print("=================", file=sys.stderr)
    print("ABCD distribution", file=sys.stderr)
    print("-----------------", file=sys.stderr)
    print(f"a+ = {len(aplus)}, {len(aplus)*100/all:4.1f}%", file=sys.stderr)
    print(f"a  = {len(aa)}, {len(aa)*100/all:4.1f}%", file=sys.stderr)
    print(f"b  = {len(bb)}, {len(bb)*100/all:4.1f}%", file=sys.stderr)
    print(f"c  = {len(cc)}, {len(cc)*100/all:4.1f}%", file=sys.stderr)
    print(f"d  = {len(d)}, {len(d)*100/all:4.1f}%", file=sys.stderr)

def print_distribution(scores):
    maxscores = max(scores)
    numinterval = maxscores // 10 + 1
    counts = np.zeros(numinterval, dtype=np.int)
    for c in scores:
        cat = c // 10
        counts[cat] += 1
    print("==================", file=sys.stderr)
    print("Score distribution", file=sys.stderr)
    print("------------------", file=sys.stderr)
    print("L.score: num:", file=sys.stderr)
    maxcount = max(counts)
    if maxcount > 80:
        unit = 80.0/maxcount
    else:
        unit = 1.0
    for i in range(numinterval):
        cat = numinterval - i - 1
        print(f"{10*cat:5}- :{counts[cat]:4}: ", end="", file=sys.stderr)
        for x in range(int(counts[cat]*unit)):
            print("*", end="", file=sys.stderr)
        print("", file=sys.stderr)

#### join

def print_only_ids(df, idlabel, ncol):
    num = 0
    for i in df[idlabel]:
        if num == 0:
            print("    ", end="", file=sys.stderr)
        elif num%ncol == 0:
            print(", \n    ", end="", file=sys.stderr)
        else:
            print(", ", end="", file=sys.stderr)
        print(f"{i}", end="", file=sys.stderr)
        num += 1
    print("", file=sys.stderr)

def join(id_scores, joinfilename, how):
#    id_scores_join = pd.read_csv(joinfilename, header=None, dtype=int, skipinitialspace=True)
    id_scores_join = pd.read_csv(joinfilename, header=None, dtype=object, skipinitialspace=True)
    id_scores_join.fillna('0', inplace=True)
    id_scores_join = id_scores_join.astype('int')
    new_id_scores = pd.merge(id_scores, id_scores_join, on=0, how=how)
    outer_id_scores = pd.merge(id_scores, id_scores_join, on=0, how='outer', indicator='from')
    nrow_left = id_scores.shape[0]
    nrow_right = id_scores_join.shape[0]
    nrow_new = new_id_scores.shape[0]
    nrow_outer = outer_id_scores.shape[0]
    print(f"Join({how}): left({nrow_left}) + right({nrow_right}) = {how}-join({nrow_new})", file=sys.stderr)
    left_only = outer_id_scores[outer_id_scores['from']=='left_only']
    right_only = outer_id_scores[outer_id_scores['from']=='right_only']
    print(f"  #left_only = {left_only.shape[0]}: keep left scores", file=sys.stderr)
    if left_only.shape[0] > 0:
        print_only_ids(left_only, 0, 5)
    if how == 'left':
        print(f"  #right_only = {right_only.shape[0]}: ignored by 'left-join'", file=sys.stderr)
    else:
        print(f"  #right_only = {right_only.shape[0]}: keep right scores", file=sys.stderr)
    if right_only.shape[0] > 0:
        print_only_ids(right_only, 0, 5)
    scores_sum = new_id_scores.iloc[:,1:].fillna(0).apply(sum, axis=1, raw=True)
    joined_new_id_scores = pd.concat([new_id_scores.iloc[:,0], scores_sum], axis=1, ignore_index=True)
    joined_new_id_scores.fillna(0, inplace=True)
#    joined_new_id_scores.astype(int, inplace=True) # inplace optoin is ineffective
    joined_new_id_scores = joined_new_id_scores.astype(int)
    return joined_new_id_scores

def twinsjoin(twins, id_scores, joinfilename):
    del twins['総合評価']
    id_scores.columns=['学籍番号', '総合評価']
    newtwins = pd.merge(twins, id_scores, on='学籍番号', how='left')
    # check correctness
    twins_outer = pd.merge(twins, id_scores, on='学籍番号', how='outer', indicator='from')
    left_only = twins_outer[twins_outer['from']=='left_only']
    right_only = twins_outer[twins_outer['from']=='right_only']
    if left_only.shape[0] > 0 or right_only.shape[0] > 0:
        print("WARNING!!: occur something wrongs in 'twinsjoin'", file=sys.stderr)
        print("WARNING!!: occur something wrongs in 'twinsjoin'", file=sys.stderr)
    """
    nrow_left = twins.shape[0]
    nrow_right = id_scores.shape[0]
    nrow_new = newtwins.shape[0]
    nrow_outer = twins_outer.shape[0]
    print(f"Join(for Twins file): left({nrow_left}) + right({nrow_right}) = LEFT-join({nrow_new})", file=sys.stderr)
    print(f"  #left_only = {left_only.shape[0]}: keep twins scores (or put a zero score)", file=sys.stderr)
    if left_only.shape[0] > 0:
        print_only_ids(left_only, '学籍番号', 5)
    print(f"  #right_only = {right_only.shape[0]}: ignored", file=sys.stderr)
    if right_only.shape[0] > 0:
        print_only_ids(right_only, '学籍番号', 5)
    """
    newtwins['総合評価'].fillna(0, inplace=True)
    newscores = newtwins['総合評価'].astype('int')
    del newtwins['総合評価']
    newtwins = pd.concat([twins, newscores], axis=1, ignore_index=True)
    newtwins.columns=['科目番号', '学籍番号', '学期区分', '学期評価', '総合評価']
    new_id_scores = newtwins[['学籍番号', '総合評価']]
    return newtwins, new_id_scores

#### record
def record(meibofilename, csvfilename2s):
    df = read_meibo(meibofilename)
    df.rename(columns={'学籍番号':0}, inplace=True)
    for csvfilename2 in csvfilename2s:
        df2 = pd.read_csv(csvfilename2, header=None, skipinitialspace=True)
        df = pd.merge(df, df2, on=0, how='outer')
    df.rename(columns={0:'学籍番号'}, inplace=True)
    df = df.sort_values(by=['所属学類','学籍番号'], ascending=True)
    return df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='support tools of scoring for performance evaluation', prog='score')
    parser.add_argument('csvfile')
    parser.add_argument('-marksheet', nargs=2, default=None, metavar=('ref', 'desired_pscore'))
    parser.add_argument('-crate', action='store_true', default=False)
    parser.add_argument('-join', default=None, metavar='csvfile2')
    parser.add_argument('-record', nargs='+', default=None, metavar=('csvfile2'))
    parser.add_argument('-twins', action='store_true', default=False)
    parser.add_argument('-adjust', nargs=3, type=float, default=None, metavar=('x', 'y', 'xmax'))
    parser.add_argument('-a2djust', nargs=4, type=float, default=None, metavar=('A+', 'A', 'B', 'C'))
    parser.add_argument('-interval', nargs=2, type=int, default=None, metavar=('min', 'max'))
    parser.add_argument('-distribution', action='store_true', default=False)
    parser.add_argument('-abcd', action='store_true', default=False)
    parser.add_argument('-statistics', action='store_true', default=False)
    parser.add_argument('-gakuruistat', default=None, metavar='csv-meibo-utf8')
    parser.add_argument('-nostdout', action='store_true', default=False)
    parser.add_argument('-output', default=None, metavar='filename')

    args = parser.parse_args()

    if args.marksheet and args.twins:
        print("scoring error: exclusive options: -marksheet and -twins", file=sys.stderr)
        exit()

    if args.record and args.twins:
        print("scoring error: exclusive options: -record and -twins", file=sys.stderr)
        exit()

    if args.record:
        print("NOTICE:", file=sys.stderr)
        print("-record option ignores all other options but -output option", file=sys.stderr)
        df = record(args.csvfile, args.record)
        if args.output:
            df.to_excel(args.output, index=False)
        else:
            df.to_csv(sys.stdout, index=False)
        exit()

    if args.marksheet:
        QuestionReferences = eval(open(args.marksheet[0]).read())
        id_scores = marksheetScoring(args.csvfile, args.crate, int(args.marksheet[1]))
    else:
        if args.twins:
            id_scores, twins = read_twins_upload_file(args.csvfile)
        else:
            # id_scores = pd.read_csv(args.csvfile, header=None, dtype=int, skipinitialspace=True)
            id_scores = pd.read_csv(args.csvfile, header=None, dtype=object, skipinitialspace=True)
            id_scores.fillna('0', inplace=True)
            id_scores = id_scores.astype('int')

    if args.join:
        if args.twins:
            id_scores = join(id_scores, args.join, 'left')
        else:
            id_scores = join(id_scores, args.join, 'outer')

    if args.adjust:
        id_scores = adjust(id_scores, args.adjust)
    if args.a2djust:
        id_scores = a2djust(id_scores, args.a2djust)
    if args.interval:
        id_scores = interval(id_scores, args.interval)

    if args.twins:
        twins, id_scores = twinsjoin(twins, id_scores, args.join)

    if args.statistics:
        print_stat(id_scores.iloc[:,1])
    if args.abcd:
        print_abcd(id_scores.iloc[:,1])
    if args.gakuruistat:
        print_stat_gakurui(id_scores, args.gakuruistat)
    if args.distribution:
        print_distribution(id_scores.iloc[:,1])

    if not args.nostdout or args.output:
        if args.output:
            output = args.output
        else:
            output = sys.stdout
        if args.twins:
            twins.to_csv(output, index=False, encoding='cp932')
        else:
            id_scores.to_csv(output, index=False, header=False)

