# from https://github.com/thuhcsi/SpanPSP

import re


def add_seg(a):
    a = re.sub(u"[\u4e00-\u9fa5]+", '*', a)
    a = re.sub(r'[a-zA-Z]+', '*', a)
    a = a.replace('(* *)', 'W')
    a = re.sub(r'[^0-9A-Za-z]+', '', a)
    return a


def remove_top(a):
    a = a.replace('(TOP (S ', '')
    a = a[::-1].replace('))', '', 1)[::-1]
    return a


def replace_n(a, i, j, num):
    a = a.replace(i, '*', num)
    a = a.replace('*', i, num - 1)
    a = a.replace('*', j)
    return a


def replace1(a):
    a = re.sub('\n', '', a)
    for i in range(len(a)):
        num_left = 0
        num_right = 0
        flag = 0
        for j in range(len(a)):
            if a[j] == '(':
                num_left += 1
                if a[j + 1] == 'S' and flag == 0:
                    b = a[j + 1]
                    flag = 1
                if a[j + 1] == '#' and flag == 0:
                    b = a[j + 1] + a[j + 2]
                    flag = 1
                    # print('mmmmm',b)
            elif a[j] == ')':
                num_right += 1
                if num_right == num_left and a[j - 1] == ')':
                    # print(num_right, b)
                    # print('mmmmmmm:',a)
                    a = replace_n(a, ')', b, num_left)
                    a = a.replace('(' + b, '', 1)
                    break
    return a


punctuation_list = ['，', '。', '、', '；', '：', '？', '！', '“', '”', '‘', '’', '—', '…', '（', '）', '《', '》']


def data_pre_processing(x):
    x = re.sub('——', '—', x)
    x = re.sub('……', '…', x)
    return x


def separate_each_character(x):
    '''
    input:  猴子#2用#1尾巴#2荡秋千#3。
    output: (n 猴)(n 子)2(n 用)1(n 尾)(n 巴)2(n 荡)(n 秋)(n 千)3(。 。)
    '''
    x = re.sub('#', '', x)
    x_list = []
    for i in x:
        if i in ['1', '2', '3']:
            x_list.append(i)
        elif i in punctuation_list:
            i = '(' + i + ' ' + i + ')'
            x_list.append(i)
        else:
            i = '(' + 'n' + ' ' + i + ')'
            x_list.append(i)

    x = ''.join(x_list)
    return x


def seq2tree(x):
    '''
    input:  (n 猴)(n 子)2(n 用)1(n 尾)(n 巴)2(n 荡)(n 秋)(n 千)3(。 。)
    output: (TOP (S (#3 (#2 (#1 (n 猴)(n 子)))(#2 (#1 (n 用))(#1 (n 尾)(n 巴)))(#2 (#1 (n 荡)(n 秋)(n 千))))(。 。)))
    '''
    iph_list = x.split('3')
    iph_ = []
    for iph in iph_list[:-1]:
        pph_list = iph.split('2')
        pph_ = []
        for pph in pph_list:
            pw_list = pph.split('1')
            pw_ = []
            for pw in pw_list:
                pw = '(' + '#1' + ' ' + pw + ')'
                pw_.append(pw)
            pw_ = ''.join(pw_)
            pw_ = '(' + '#2' + ' ' + pw_ + ')'
            pph_.append(pw_)
        pph_ = ''.join(pph_)
        pph_ = '(' + '#3' + ' ' + pph_ + ')'
        iph_.append(pph_)
    iph_.append(iph_list[-1])
    iph_ = ''.join(iph_)
    tree = '(' + 'TOP' + ' ' + '(' + 'S' + ' ' + iph_ + ')' + ')'
    return tree


def convert_to_W(line):
    line = data_pre_processing(line.strip())
    line = separate_each_character(line)
    line = seq2tree(line)

    ss = line
    sss = remove_top(ss)
    sss = replace1(sss)
    ssss = add_seg(sss)

    return ssss


def convert_to_01(line):
    s0 = line
    s = re.sub('\n', '', s0)

    compileX = re.compile(r'\d+')
    num_result = compileX.findall(s)
    for i in num_result:
        s = re.sub(i, max(i), s, 1)

    s = re.sub('W', '0', s)
    s = re.sub('01', '1', s)
    s = re.sub('02', '2', s)
    s = re.sub('03', '3', s)

    return s


def score(TP, FP, FN):
    if TP + FP == 0:
        precision = 0.01
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0.01
    else:
        recall = TP / (TP + FN)
    if precision + recall ==0:
      f1score = 0.01
    else:
      f1score = 2 * precision * recall / (precision + recall)
    return precision*100, recall*100, f1score*100


def evaluate(test_sen_list, predicted_sen_list):
    # a12: test 1, predicted 2
    a00 = a01 = a02 = a03 = a10 = a11 = a12 = a13 = a20 = a21 = a22 = a23 = a30 = a31 = a32 = a33 = 0
    num = 0
    num_match_sen = 0
    for i in range(len(test_sen_list)):
        tW = convert_to_W(test_sen_list[i])
        pW = convert_to_W(predicted_sen_list[i])
        t = convert_to_01(tW)
        p = convert_to_01(pW)

        if t == p:
            num_match_sen += 1

        # print(num, '\n', t, test_s00_list[i], test_s0_list[i], '\n', p ,predicted_s00_list[i], predicted_s0_list[i])
        print(f'Testing sample {i}')
        print('\tGround Truth:', test_sen_list[i])
        print('\tPrediction  :', predicted_sen_list[i])
        print('\tGround Truth:', tW)
        print('\tPrediction  :', pW)
        print('\tGround Truth:', t.strip().replace(' ', ''))
        print('\tPrediction  :', p.strip().replace(' ', ''))

        if len(t) != len(p):
            num += 1
        else:
            for j in range(len(t)):
                if t[j] == '0':
                    if p[j] == '0':
                        a00 += 1
                    if p[j] == '1':
                        a01 += 1
                    if p[j] == '2':
                        a02 += 1
                    if p[j] == '3':
                        a03 += 1
                if t[j] == '1':
                    if p[j] == '0':
                        a10 += 1
                    if p[j] == '1':
                        a11 += 1
                    if p[j] == '2':
                        a12 += 1
                    if p[j] == '3':
                        a13 += 1
                if t[j] == '2':
                    if p[j] == '0':
                        a20 += 1
                    if p[j] == '1':
                        a21 += 1
                    if p[j] == '2':
                        a22 += 1
                    if p[j] == '3':
                        a23 += 1
                if t[j] == '3':
                    if p[j] == '0':
                        a30 += 1
                    if p[j] == '1':
                        a31 += 1
                    if p[j] == '2':
                        a32 += 1
                    if p[j] == '3':
                        a33 += 1

    # https://github.com/thuhcsi/SpanPSP/issues/3
    # https://github.com/thuhcsi/SpanPSP/commit/b04d57f9269c9190ce975d63dc6d6aad56ca934a
    # precision1, recall1, fscore1 = score(a11 + a12 + a13 + a21 + a22 + a23 + a31 + a32 + a33, a01 , a10 + a20 + a30)

    precision1, recall1, fscore1 = score(
        TP=a11 + a12 + a13 + a21 + a22 + a23 + a31 + a32 + a33,
        FP=a01 + a02 + a03,
        FN=a10 + a20 + a30
    )
    precision2, recall2, fscore2 = score(
        TP=a22 + a23 + a32 + a33,
        FP=a02 + a03 + a12 + a13,
        FN=a20 + a21 + a30 + a31
    )
    precision3, recall3, fscore3 = score(
        TP=a33,
        FP=a03 + a13 + a23,
        FN=a30 + a31 + a32
    )
    precision = float((precision1 + precision2 + precision3) / 3)
    recall = float((recall1 + recall2 + recall3) / 3)
    fscore = float((fscore1 + fscore2 + fscore3) / 3)

    completematch = float(100 * num_match_sen / len(test_sen_list))

    print('\n| Level | Precision | Recall | F-Score |')
    print('| ----- | --------- | ------ | ------- |')
    print(f'|PW  #1 | {round(precision1, 2)} | {round(recall1, 2)} | {round(fscore1, 2)} |')
    print(f'|PPH #2 | {round(precision2, 2)} | {round(recall2, 2)} | {round(fscore2, 2)} |')
    print(f'|IPH #3 | {round(precision3, 2)} | {round(recall3, 2)} | {round(fscore3, 2)} |')
    print(f'|Average | {round(precision, 2)} | {round(recall, 2)} | {round(fscore, 2)} |')
    print(f'Exact Match: {num_match_sen} / {len(test_sen_list)} = {completematch}%')

    return recall, precision, fscore, completematch


def get_sentence_list(file_path):
    sentences = open(file_path, encoding='utf-8').readlines()
    print(f'loading {file_path}, found {len(sentences)} sentences')
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        results.append(sentence)
    print(results)
    return results
