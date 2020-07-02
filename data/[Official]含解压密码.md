### [Official] Schedule, Passwords, Script for Evaluation, and FAQ!

------

## Schedule

**[Important]** Your ndcg@50_full for phase 7,8,9 need to be higher than or equal to **rank 65** in order to be qualified.

- 2020-06-04 10:30: There are about 650 teams who have submitted valid predictions, i.e. scores > 0, for phase 0-6. You therefore need to be the top 65 teams to be in the top 10%.

Phase 7,8,9 are postponed from May 22-Jun 11 to Jun 5-Jun 11. We make the change to the schedule because:

- We received tons of emails that express concerns about cheating and ask us to NOT show the scores when we are in the final stage (phase 7,8,9). And we agreed with the proposal and decided to not show the scores after June 5.
- Many people say that they are not ready yet and want to keep improving their results on the development set (phase 0-6).

Note that the final leaderboard will take only phase T=7,8,9 into account when deciding the winning teams. That is, dataset A = phase 0,1,…,6, while dataset B = 7,8,9. Three submissions per day for dataset A, and one submission per day for dataset B.

| Phase | Start time (UNIX timestamp) | Start time (GMT+8)  | Does it affect the final leaderboard? |
| :---- | :-------------------------- | :------------------ | :------------------------------------ |
| T=0   | 0                           | 1970-01-01 08:00:00 | No                                    |
| T=1   | 1586534399                  | 2020-04-10 23:59:59 | No                                    |
| T=2   | 1587139199                  | 2020-04-17 23:59:59 | No                                    |
| T=3   | 1587743999                  | 2020-04-24 23:59:59 | No                                    |
| T=4   | 1588348799                  | 2020-05-01 23:59:59 | No                                    |
| T=5   | 1588953599                  | 2020-05-08 23:59:59 | No                                    |
| T=6   | 1589558399                  | 2020-05-15 23:59:59 | No                                    |
| T=7   | 1591315200                  | 2020-06-05 08:00:00 | Yes                                   |
| T=8   | 1591315200                  | 2020-06-05 08:00:00 | Yes                                   |
| T=9   | 1591315200                  | 2020-06-05 08:00:00 | Yes                                   |

## Passwords for Unzipping

Password Filename:

```json
7c2d2b8a636cbd790ff12a007907b2ba underexpose_train_click-1
ea0ec486b76ae41ed836a8059726aa85 underexpose_train_click-2
65255c3677a40bf4d341b0c739ad6dff underexpose_train_click-3
c8376f1c4ed07b901f7fe5c60362ad7b underexpose_train_click-4
63b326dc07d39c9afc65ed81002ff2ab underexpose_train_click-5
f611f3e477b458b718223248fd0d1b55 underexpose_train_click-6
ec191ea68e0acc367da067133869dd60 underexpose_train_click-7
90129a980cb0a4ba3879fb9a4b177cd2 underexpose_train_click-8
f4ff091ab62d849ba1e6ea6f7c4fb717 underexpose_train_click-9

96d071a532e801423be614e9e8414992 underexpose_test_click-1
503bf7a5882d3fac5ca9884d9010078c underexpose_test_click-2
dd3de82d0b3a7fe9c55e0b260027f50f underexpose_test_click-3
04e966e4f6c7b48f1272a53d8f9ade5d underexpose_test_click-4
13a14563bf5528121b8aaccfa7a0dd73 underexpose_test_click-5
dee22d5e4a7b1e3c409ea0719aa0a715 underexpose_test_click-6
69416eedf810b56f8a01439e2061e26d underexpose_test_click-7
55588c1cddab2fa5c63abe5c4bf020e5 underexpose_test_click-8
caacb2c58d01757f018d6b9fee0c8095 underexpose_test_click-9
```



## [Must Read!] Official Script for Evaluation

```python
# coding=utf-8
from __future__ import division
from __future__ import print_function

import datetime
import json
import sys
import time
from collections import defaultdict

import numpy as np

# the higher scores, the better performance
def evaluate_each_phase(predictions, answers):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        while rank < 50 and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)

# submit_fname is the path to the file submitted by the participants.
# debias_track_answer.csv is the standard answer, which is not released.
def evaluate(stdout, submit_fname,
             answer_fname='debias_track_answer.csv', current_time=None):
    schedule_in_unix_time = [
        0,  # ........ 1970-01-01 08:00:00 (T=0)
        1586534399,  # 2020-04-10 23:59:59 (T=1)
        1587139199,  # 2020-04-17 23:59:59 (T=2)
        1587743999,  # 2020-04-24 23:59:59 (T=3)
        1588348799,  # 2020-05-01 23:59:59 (T=4)
        1588953599,  # 2020-05-08 23:59:59 (T=5)
        1589558399,  # 2020-05-15 23:59:59 (T=6)
        1591315200,  # 2020-06-05 08:00:00 (Beijing Time) (T=7)
        1591315200,  # 2020-06-05 08:00:00 (Beijing Time) (T=8)
        1591315200  # .2020-06-05 08:00:00 (Beijing Time) (T=9)
    ]
    assert len(schedule_in_unix_time) == 10

    if current_time is None:
        current_time = int(time.time())
    print('current_time:', current_time)
    print('date_time:', datetime.datetime.fromtimestamp(current_time))
    current_phase = 0
    while (current_phase < 9) and (
            current_time > schedule_in_unix_time[current_phase + 1]):
        current_phase += 1
    print('current_phase:', current_phase)

    try:
        answers = [{} for _ in range(10)]
        with open(answer_fname, 'r') as fin:
            for line in fin:
                line = [int(x) for x in line.split(',')]
                phase_id, user_id, item_id, item_degree = line
                assert user_id % 11 == phase_id
                # exactly one test case for each user_id
                answers[phase_id][user_id] = (item_id, item_degree)
    except Exception as _:
        return report_error(stdout, 'server-side error: answer file incorrect')

    try:
        predictions = {}
        with open(submit_fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(',')
                user_id = int(line[0])
                if user_id in predictions:
                    return report_error(stdout, 'submitted duplicate user_ids')
                item_ids = [int(i) for i in line[1:]]
                if len(item_ids) != 50:
                    return report_error(stdout, 'each row need have 50 items')
                if len(set(item_ids)) != 50:
                    return report_error(
                        stdout, 'each row need have 50 DISTINCT items')
                predictions[user_id] = item_ids
    except Exception as _:
        return report_error(stdout, 'submission not in correct format')

    scores = np.zeros(4, dtype=np.float32)

    # The final winning teams will be decided based on phase T=7,8,9 only.
    # We thus fix the scores to 1.0 for phase 0,1,2,...,6 at the final stage.
    if current_phase >= 7:  # if at the final stage, i.e., T=7,8,9
        scores += 7.0  # then fix the scores to 1.0 for phase 0,1,2,...,6
    phase_beg = (7 if (current_phase >= 7) else 0)
    phase_end = current_phase + 1
    for phase_id in range(phase_beg, phase_end):
        for user_id in answers[phase_id]:
            if user_id not in predictions:
                return report_error(
                    stdout, 'user_id %d of phase %d not in submission' % (
                        user_id, phase_id))
        try:
            # We sum the scores from all the phases, instead of averaging them.
            scores += evaluate_each_phase(predictions, answers[phase_id])
        except Exception as _:
            return report_error(stdout, 'error occurred during evaluation')

    return report_score(
        stdout, score=float(scores[0]),
        ndcg_50_full=float(scores[0]), ndcg_50_half=float(scores[1]),
        hitrate_50_full=float(scores[2]), hitrate_50_half=float(scores[3]))

# FYI. You can create a fake answer file for validation based on this. For example,
# you can mask the latest ONE click made by each user in underexpose_test_click-T.csv,
# and use those masked clicks to create your own validation set, i.e.,
# a fake underexpose_test_qtime_with_answer-T.csv for validation.
def _create_answer_file_for_evaluation(answer_fname='debias_track_answer.csv'):
    train = 'underexpose_train_click-%d.csv'
    test = 'underexpose_test_click-%d.csv'

    # underexpose_test_qtime-T.csv contains only <user_id, time>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    answer = 'underexpose_test_qtime_with_answer-%d.csv'  # not released

    item_deg = defaultdict(lambda: 0)
    with open(answer_fname, 'w') as fout:
        for phase_id in range(10):
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(test % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    assert user_id % 11 == phase_id
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)
```

If you find any potential bugs in this script, please contact the organizer via email: [kdd_cup@alibaba-inc.com](mailto:kdd_cup@alibaba-inc.com)

## FAQ

- **Q: Should our submitted file include our predictions for all underexpose_test_qtime-0,1,2,…,T.csv when we're in phase T? Or just underexpose_test_qtime-T.csv?**
- A: Please make sure that your have read the official script for evaluation. In short, you should include your predictions for all underexpose_test_click-0,1,2,…,T when in phase T. Note that we have ensured that each user_id will not appear in more than one phase. Therefore, you don't need to specify which phase each row of your submission is for. Each row of your submission should contain just one user_id, followed by 50 different item_ids.
- **Q: Why can't I find underexpose_test_qtime_with_answer-T.csv?**
- A: underexpose_test_qtime_with_answer-T.csv is the standard answer, which is of course not released. However, we do have released underexpose_test_qtime-T.csv, which contains the first column (user_id) and the third column (time) of underexpose_test_qtime_with_answer-T.csv.
- **Q: Can I use underexpose_train_click-9.csv and underexpose_test_click-9.csv as extra data, in addition to underexpose_train_click-7.csv, when making predictions for underexpose_test_qtime-7.csv?**
- A: Yes. You can always improve your predictions for phase T, even if we are already in phase T' > T.
- **Q: What data source? And what bias to reduce?**
- A: The data source is the click data in mobile taobao's recommendation system. The sampling procedure is almost totally random, except that we have added a few constraints. For example, each user or item should have at least more than X clicks if you look at the complete dataset, which involves many days. However, do note that an item can have even zero clicks until some day. We are focusing on the *exposure bias* in E-Comm RecSys. Not all bias in all applications, which is a goal impossible to achieve. Note that the data distribution may not be stable, though, since it is collected across many days.
- **Q: Why is the timestamp in (0.0, 1.0)?**
- A: The timestamp provided in the dataset is actually time=(unix_timestamp - random_number_1) / random_number_2. These two random numbers are kept the same for all clicks. The time orders of the clicks are therefore preserved.
- **Q: Maximum number of members per team?**
- A: Ten at the moment.
- **Q: How to win?**
- A: A team first need to be among the top 10% in terms of NDCG@50-full (aka. ndcg_50_full on the leaderboard) so as to be *qualified*. The winning teams will then be the *qualified* teams that achieve the best NDCG@50-rare (aka. ndcg_50_half on the leaderboard).
- **Q: Are the teams who have registered but have no submissions for dataset B counted when deciding the top 10%?**
- A: No. FYI, the number of qualified teams will be set to MAX(10, the number of teams among top 10%). *We will announce the exact rank you need to achieve in order to be qualified when phase 9 begins.*
- **Q: Are there overlapped (duplicate) data between phase T and phase T+1? Should I remove the duplicate data?**
- A: Yes. Roughly 2/3 data will be the same. You may need to remove them if you need to use more than two phases when making predictions for the present phase.
- **Q: If I score a very high NDCG@50-full that gets me into the top 10% but my NDCG@50-rare is pretty low. Then I make a trade-off b/w these 2 metrics and my new result scores a lower NDCG@50-FULL, which still gets me into the top 10%, but with a much higher NDCG@50-rare. Which one of these 2 results will be used as the final submission, since the leaderboard only shows the first one?**
- A: We only look at the results shown on the leaderboard. We won't change the leaderboard so that it sorts the scores by ndcg_50_half directly, because this design of sorting by ndcg_50_full is intended as a way to encourage you to *only submit one time in total for phase 9*. If you still need to adjust your algorithm when we are already in phase 9, then you are probably just overfitting the dataset. P.S.: Tianchi doesn't support limiting the total submit times directly at the moment. It only supports limiting the submit times *for each day*.