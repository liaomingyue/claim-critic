# 错误操作卡片

造错用的错误体系：四层二十种操作，每种操作一张卡片。给错误合成模型出题时，把对应的卡片整张写进提示词。
这一页由 `python -m scripts.dump_cards` 从代码（`ragtruth/error_ops.py`）生成，卡片的原文是英文，因为它直接进提示词。

分层的依据是“错的东西从哪来”：

| 层 | 含义 | 操作 |
|---|---|---|
| 值 | 某个成分本身被换掉或添上 | [近义替换](#swap_term)、[改数字](#change_number)、[加事实](#add_detail)、[说成相反](#flip_outcome) |
| 挂接 | 成分都在原文里，但安错了地方 | [换说话人](#change_speaker)、[修饰语挂错](#move_modifier)、[方向倒置](#reverse_direction)、[所指换对象](#swap_referent)、[所属关系换人](#change_owner)、[数量挂错](#move_number)、[时间地点挂错](#move_time_place)、[两处并成一句](#merge_events) |
| 限定 | 确定程度、范围、条件变了 | [删改推测词](#drop_hedge)、[加推测词](#add_hedge)、[删掉「据称」](#drop_reported)、[打算说成已发生](#plan_as_done)、[范围放大](#generalize)、[删条件或限制](#drop_restriction) |
| 句间 | 句子之间的关系，或凭空加上的推论 | [加或换连接词](#add_link)、[加结论或评价](#add_conclusion) |

每张卡片有五部分：

- **做法**：这种操作对句子做什么。
- **适用条件**：句子要有什么才做得成。没有数字的句子造不出数字错，这类任务出题前就筛掉。
- **为什么算错**：改完之后，句子和原文哪里对不上。
- **怎么检查**：核实时看什么。
- **例子**：两对真实的句子，一句忠于原文，一句是模型自己写错的，两句只差这一处。

## 值层

某个成分本身被换掉或添上。

<a id="swap_term"></a>

### 近义替换（`swap_term`）

| | |
|---|---|
| 做法 | replace one word or short phrase with a similar one that refers to something different (a neighbouring category, a stronger or weaker act), changing nothing else |
| 适用条件 | the sentence has a specific noun, verb or adjective that has a close neighbour with a different meaning |
| 为什么算错 | the new term refers to something the source does not say, although it sounds like the same thing |
| 怎么检查 | 找到原文里对应的那个词，看换上的词所指是不是同一样东西 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Replace 'friends' with 'family members'.
- Replace 'nominated for eviction' with 'evict'.
- Replace 'the people of Latin America' with 'Latin American leaders'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Tristan da Cunha, a remote archipelago in the South Atlantic with just under 270 residents, is holding an international design competition to modernize its aging infrastructure and ensure its sustainability as it approaches the 200th anniversary of its settlement. | Tristan da Cunha, a remote archipelago in the South Atlantic with just under 270 residents, is seeking an international design competition to modernize its aging infrastructure and ensure its sustainability as it approaches the 200th anniversary of its settlement. |
| As fit golfers like Tiger Woods, Adam Scott, and Jordan Spieth win the Masters, there is a growing emphasis on functional movement to improve swing power and reduce injury risk. | As fit golfers like Tiger Woods, Adam Scott, and Jordan Spieth dominate the Masters, there is a growing emphasis on functional movement to improve swing power and reduce injury risk. |

<details><summary>写进提示词的卡片原文</summary>

```
replace one word or short phrase with a similar one that refers to something different (a neighbouring category, a stronger or weaker act), changing nothing else.
Where it can be done: the sentence has a specific noun, verb or adjective that has a close neighbour with a different meaning.
Why the result is an error: the new term refers to something the source does not say, although it sounds like the same thing.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Tristan da Cunha, a remote archipelago in the South Atlantic with just under 270 residents, is holding an international design competition to modernize its aging infrastructure and ensure its sustainability as it approaches the 200th anniversary of its settlement.
  With the error: Tristan da Cunha, a remote archipelago in the South Atlantic with just under 270 residents, is seeking an international design competition to modernize its aging infrastructure and ensure its sustainability as it approaches the 200th anniversary of its settlement.
- Faithful: As fit golfers like Tiger Woods, Adam Scott, and Jordan Spieth win the Masters, there is a growing emphasis on functional movement to improve swing power and reduce injury risk.
  With the error: As fit golfers like Tiger Woods, Adam Scott, and Jordan Spieth dominate the Masters, there is a growing emphasis on functional movement to improve swing power and reduce injury risk.
```

</details>

<a id="change_number"></a>

### 改数字（`change_number`）

| | |
|---|---|
| 做法 | change a number, date, duration or count to a different value, or turn a count into a proportion or the reverse |
| 适用条件 | the sentence states a number, date, duration or count |
| 为什么算错 | the source gives a different value for the same thing |
| 怎么检查 | 在原文里找到这个数，核对数值和单位 |
| 出题前的规则 | 句子里有数字、日期或数量词 |

改法举例：

- Change 'around 15' to 'over 15'.
- Change the count '55 Americans' into '55% American'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Obesity is a widespread issue, with 70% of adults overweight or obese and rising rates among children. | Obesity is a widespread issue, with over 70% of adults overweight or obese and rising rates among children. |
| The Italian Coast Guard reported nearly 10,000 migrants have arrived on Italian shores since the weekend, with spring and calmer seas not stopping the flow. | The Italian Coast Guard reported over 10,000 migrants have arrived on Italian shores since the weekend, with spring and calmer seas not stopping the flow. |

<details><summary>写进提示词的卡片原文</summary>

```
change a number, date, duration or count to a different value, or turn a count into a proportion or the reverse.
Where it can be done: the sentence states a number, date, duration or count.
Why the result is an error: the source gives a different value for the same thing.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Obesity is a widespread issue, with 70% of adults overweight or obese and rising rates among children.
  With the error: Obesity is a widespread issue, with over 70% of adults overweight or obese and rising rates among children.
- Faithful: The Italian Coast Guard reported nearly 10,000 migrants have arrived on Italian shores since the weekend, with spring and calmer seas not stopping the flow.
  With the error: The Italian Coast Guard reported over 10,000 migrants have arrived on Italian shores since the weekend, with spring and calmer seas not stopping the flow.
```

</details>

<a id="add_detail"></a>

### 加事实（`add_detail`）

| | |
|---|---|
| 做法 | add or substitute one specific detail that the source does not contain (what happened next, where, what kind), stated as plainly as the rest of the sentence |
| 适用条件 | the sentence states a fact that could be made more specific |
| 为什么算错 | the added detail is nowhere in the source |
| 怎么检查 | 在原文里找这个细节，找不到就是错 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Replace 'was discharged' with 'resumed work'.
- Add a clause saying some returned to their villages.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| A 24-year-old man, Sid Ahmed Ghlam, was taken into custody after calling for an ambulance, only to be found with weapons, ammunition, and evidence of plots to target churches. | A 24-year-old French man, Sid Ahmed Ghlam, was taken into custody after calling for an ambulance, only to be found with weapons, ammunition, and evidence of plots to target churches. |
| Muhammadu Buhari, a former military leader, won Nigeria's presidency, defeating incumbent Goodluck Jonathan by about 2 million votes in a democratic election. | Muhammadu Buhari, a former military leader, won Nigeria's presidency, defeating incumbent Goodluck Jonathan by about 2 million votes in a peaceful democratic election. |

<details><summary>写进提示词的卡片原文</summary>

```
add or substitute one specific detail that the source does not contain (what happened next, where, what kind), stated as plainly as the rest of the sentence.
Where it can be done: the sentence states a fact that could be made more specific.
Why the result is an error: the added detail is nowhere in the source.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: A 24-year-old man, Sid Ahmed Ghlam, was taken into custody after calling for an ambulance, only to be found with weapons, ammunition, and evidence of plots to target churches.
  With the error: A 24-year-old French man, Sid Ahmed Ghlam, was taken into custody after calling for an ambulance, only to be found with weapons, ammunition, and evidence of plots to target churches.
- Faithful: Muhammadu Buhari, a former military leader, won Nigeria's presidency, defeating incumbent Goodluck Jonathan by about 2 million votes in a democratic election.
  With the error: Muhammadu Buhari, a former military leader, won Nigeria's presidency, defeating incumbent Goodluck Jonathan by about 2 million votes in a peaceful democratic election.
```

</details>

<a id="flip_outcome"></a>

### 说成相反（`flip_outcome`）

| | |
|---|---|
| 做法 | state the opposite of what the source says about an outcome, a stance or whether something happened |
| 适用条件 | the sentence states an outcome, a stance, or whether something happened |
| 为什么算错 | the source says the opposite |
| 怎么检查 | 在原文里找到这件事的结果或态度，看是不是反了 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Replace 'stabbed, in stable condition' with 'stabbed to death'.
- Flip '78 Republicans voting for expulsion' to '78 Republicans voting against his removal'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| American Etihad Airways passenger Jennifer Stewart lost her 2-year-old cat, Felix, after it went missing at New York's JFK Airport following a flight from Abu Dhabi. | American Etihad Airways passenger Jennifer Stewart lost her 2-year-old cat, Felix, after it was found loose at New York's JFK Airport following a flight from Abu Dhabi. |
| Not exactly; degenerative disc disease (ddd) and arthritis are very similar and often considered the same. | Yes, degenerative disc disease (ddd) and arthritis are very similar and often considered the same. |

<details><summary>写进提示词的卡片原文</summary>

```
state the opposite of what the source says about an outcome, a stance or whether something happened.
Where it can be done: the sentence states an outcome, a stance, or whether something happened.
Why the result is an error: the source says the opposite.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: American Etihad Airways passenger Jennifer Stewart lost her 2-year-old cat, Felix, after it went missing at New York's JFK Airport following a flight from Abu Dhabi.
  With the error: American Etihad Airways passenger Jennifer Stewart lost her 2-year-old cat, Felix, after it was found loose at New York's JFK Airport following a flight from Abu Dhabi.
- Faithful: Not exactly; degenerative disc disease (ddd) and arthritis are very similar and often considered the same.
  With the error: Yes, degenerative disc disease (ddd) and arthritis are very similar and often considered the same.
```

</details>

## 挂接层

成分都在原文里，但安错了地方。

<a id="change_speaker"></a>

### 换说话人（`change_speaker`）

| | |
|---|---|
| 做法 | attribute a statement, view or announcement to a different person or body that also appears in the source |
| 适用条件 | the sentence reports what someone said, thought or announced, and the source mentions another person or body |
| 为什么算错 | in the source the statement comes from someone else |
| 怎么检查 | 在原文里找这句话是谁说的 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Attribute the attendant's announcement and warning about symptoms to a passenger.
- Turn a USCIS representative's statement into something Jaddou observed.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Key points for the jury include evidence suggesting Hernandez lied to Patriots owner Robert Kraft about his innocence, as Kraft testified Hernandez said he was in a club when Lloyd was killed. | Key points for the jury include evidence suggesting Hernandez lied to Patriots owner Robert Kraft about his innocence, as Kraft claimed Hernandez was in a club when Lloyd was killed. |
| Viewed as a "safe haven" by foreign investors, London ranks as the second most important hub for ultra-high-net-worth homes globally, behind only New York. | Viewed as a "safe haven" by high-end buyers, London ranks as the second most important hub for ultra-high-net-worth homes globally, behind only New York. |

<details><summary>写进提示词的卡片原文</summary>

```
attribute a statement, view or announcement to a different person or body that also appears in the source.
Where it can be done: the sentence reports what someone said, thought or announced, and the source mentions another person or body.
Why the result is an error: in the source the statement comes from someone else.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Key points for the jury include evidence suggesting Hernandez lied to Patriots owner Robert Kraft about his innocence, as Kraft testified Hernandez said he was in a club when Lloyd was killed.
  With the error: Key points for the jury include evidence suggesting Hernandez lied to Patriots owner Robert Kraft about his innocence, as Kraft claimed Hernandez was in a club when Lloyd was killed.
- Faithful: Viewed as a "safe haven" by foreign investors, London ranks as the second most important hub for ultra-high-net-worth homes globally, behind only New York.
  With the error: Viewed as a "safe haven" by high-end buyers, London ranks as the second most important hub for ultra-high-net-worth homes globally, behind only New York.
```

</details>

<a id="move_modifier"></a>

### 修饰语挂错（`move_modifier`）

| | |
|---|---|
| 做法 | move a description, property, role or criticism from the thing it belongs to onto another thing mentioned in the source |
| 适用条件 | the sentence gives a description, property, role or criticism of one thing, and the source mentions another thing it could be attached to |
| 为什么算错 | in the source the description belongs to the other thing |
| 怎么检查 | 在原文里找这个描述说的是哪个对象 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Move 'in bad shape' from the boat to the man.
- Move the 'amateurish' criticism of Rubio's logo onto Clinton's launch.
- Transfer the author's Kentucky State Senate service to Rand Paul.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Greenpeace is furious over the U.S. government's decision to lift lease suspensions on drilling in the Chukchi Sea, warning that Shell could begin drilling in 100 days. | Greenpeace called the U.S. government's decision to lift lease suspensions on drilling in the Chukchi Sea "furious," warning that Shell could begin drilling in 100 days. |
| Tanka Maya Sitoula, a 40-year-old mother of four, survived a 36-hour ordeal after the 5-story building containing her apartment collapsed. | Tanka Maya Sitoula, a 40-year-old mother of four, survived a 36-hour ordeal after her 5-story apartment collapsed. |

<details><summary>写进提示词的卡片原文</summary>

```
move a description, property, role or criticism from the thing it belongs to onto another thing mentioned in the source.
Where it can be done: the sentence gives a description, property, role or criticism of one thing, and the source mentions another thing it could be attached to.
Why the result is an error: in the source the description belongs to the other thing.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Greenpeace is furious over the U.S. government's decision to lift lease suspensions on drilling in the Chukchi Sea, warning that Shell could begin drilling in 100 days.
  With the error: Greenpeace called the U.S. government's decision to lift lease suspensions on drilling in the Chukchi Sea "furious," warning that Shell could begin drilling in 100 days.
- Faithful: Tanka Maya Sitoula, a 40-year-old mother of four, survived a 36-hour ordeal after the 5-story building containing her apartment collapsed.
  With the error: Tanka Maya Sitoula, a 40-year-old mother of four, survived a 36-hour ordeal after her 5-story apartment collapsed.
```

</details>

<a id="reverse_direction"></a>

### 方向倒置（`reverse_direction`）

| | |
|---|---|
| 做法 | reverse the direction of a relation while keeping both sides: swap who did it and who it was done to, swap cause and effect, or swap giver and receiver |
| 适用条件 | the sentence states a relation with two sides: who did something to whom, a cause and its effect, a giver and a receiver |
| 为什么算错 | in the source the relation runs the other way |
| 怎么检查 | 在原文里找这个关系的两端，看谁对谁、谁因谁果 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Swap agent and patient so that Cox knocks over Hawking instead of Hawking knocking over Cox.
- Make Congress the recipient of funding instead of the body that should give funding to USCIS.
- Reverse the causal link so the rule change shortens the game, instead of the shortened game prompting the change.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| The Justice Department has obtained a preliminary injunction to stop a debt collection scheme targeting veterans and older citizens. | The Justice Department has granted a preliminary injunction to stop a debt collection scheme targeting veterans and older citizens. |
| An Alaska Airlines flight made an emergency landing in Seattle after the pilot reported unusual banging, and a ramp agent was then found inside the cargo hold. | An Alaska Airlines flight experienced an emergency landing in Seattle after a ramp agent was found inside the cargo hold. |

<details><summary>写进提示词的卡片原文</summary>

```
reverse the direction of a relation while keeping both sides: swap who did it and who it was done to, swap cause and effect, or swap giver and receiver.
Where it can be done: the sentence states a relation with two sides: who did something to whom, a cause and its effect, a giver and a receiver.
Why the result is an error: in the source the relation runs the other way.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: The Justice Department has obtained a preliminary injunction to stop a debt collection scheme targeting veterans and older citizens.
  With the error: The Justice Department has granted a preliminary injunction to stop a debt collection scheme targeting veterans and older citizens.
- Faithful: An Alaska Airlines flight made an emergency landing in Seattle after the pilot reported unusual banging, and a ramp agent was then found inside the cargo hold.
  With the error: An Alaska Airlines flight experienced an emergency landing in Seattle after a ramp agent was found inside the cargo hold.
```

</details>

<a id="swap_referent"></a>

### 所指换对象（`swap_referent`）

| | |
|---|---|
| 做法 | make a pronoun, a short referring phrase or the actor of an action point to a different person or thing from the source than the one the source means |
| 适用条件 | the sentence refers to a person or thing that the source keeps apart from another similar one |
| 为什么算错 | the source means the other person or thing |
| 怎么检查 | 在原文里找这个说法或代词指的是谁、是什么 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Resolve 'his body' to Hernandez instead of Odin Lloyd.
- Make 'send back to Earth' refer to the delivered cargo instead of the experiments and trash packed for return.
- Replace the actor 'prosecutors' with 'the court'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Despite encountering large groups of migrants at razor-wire barriers in Texas, including a hypothermia victim, Texas and Florida troopers have provided life-saving aid. | Despite encountering large groups of migrants at razor-wire barriers in Texas, including a hypothermia victim, U.S. and Florida troopers have provided life-saving aid. |
| Her project, which explores identity through photography, faced initial resistance as people didn't trust her motives. | Her project, which explores identity through photography, faced initial resistance as locals didn't trust her motives. |

<details><summary>写进提示词的卡片原文</summary>

```
make a pronoun, a short referring phrase or the actor of an action point to a different person or thing from the source than the one the source means.
Where it can be done: the sentence refers to a person or thing that the source keeps apart from another similar one.
Why the result is an error: the source means the other person or thing.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Despite encountering large groups of migrants at razor-wire barriers in Texas, including a hypothermia victim, Texas and Florida troopers have provided life-saving aid.
  With the error: Despite encountering large groups of migrants at razor-wire barriers in Texas, including a hypothermia victim, U.S. and Florida troopers have provided life-saving aid.
- Faithful: Her project, which explores identity through photography, faced initial resistance as people didn't trust her motives.
  With the error: Her project, which explores identity through photography, faced initial resistance as locals didn't trust her motives.
```

</details>

<a id="change_owner"></a>

### 所属关系换人（`change_owner`）

| | |
|---|---|
| 做法 | change whose something is: a relative, a possession, an employee or a work is given to a different person or body from the source |
| 适用条件 | the sentence says whose something is: a relative, a possession, an employee, a work |
| 为什么算错 | in the source it belongs to someone else |
| 怎么检查 | 在原文里找这个人或东西属于谁 |
| 出题前的规则 | 句子里写明了某样东西是谁的 |

改法举例：

- Turn her daughter's two children (her grandchildren) into 'her two children'.
- Change 'Copeland's driver' to 'the club's driver'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| On February morning, 36-year-old Mario Valencia stole a .30-30 rifle and ammunition from a Walmart in Marana, Arizona. | On February morning, 36-year-old Mario Valencia stole a .30-30 rifle and ammunition from a Walmart employee in Marana, Arizona. |
| Meanwhile, Ferrari's Sebastian Vettel claimed his third consecutive podium, expressing hope to close the gap on the championship lead. | Meanwhile, Red Bull's Sebastian Vettel claimed his third consecutive podium, expressing hope to close the gap on the championship lead. |

<details><summary>写进提示词的卡片原文</summary>

```
change whose something is: a relative, a possession, an employee or a work is given to a different person or body from the source.
Where it can be done: the sentence says whose something is: a relative, a possession, an employee, a work.
Why the result is an error: in the source it belongs to someone else.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: On February morning, 36-year-old Mario Valencia stole a .30-30 rifle and ammunition from a Walmart in Marana, Arizona.
  With the error: On February morning, 36-year-old Mario Valencia stole a .30-30 rifle and ammunition from a Walmart employee in Marana, Arizona.
- Faithful: Meanwhile, Ferrari's Sebastian Vettel claimed his third consecutive podium, expressing hope to close the gap on the championship lead.
  With the error: Meanwhile, Red Bull's Sebastian Vettel claimed his third consecutive podium, expressing hope to close the gap on the championship lead.
```

</details>

<a id="move_number"></a>

### 数量挂错（`move_number`）

| | |
|---|---|
| 做法 | keep a number from the source but make it count or measure a different thing |
| 适用条件 | the sentence states a number that counts or measures one particular thing |
| 为什么算错 | the number is in the source, but there it counts or measures something else |
| 怎么检查 | 在原文里找这个数，看它数的是什么 |
| 出题前的规则 | 句子里有数字 |

改法举例：

- Move the age 13 from the time of the incident to the narrator's current age.
- Reattach the 5-mile distance from crash-site-to-campus to university-to-campus.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| - Prepare the grill with two zones: high and medium-low heat (for charcoal, this is 2 to 3 coals thick for the high zone and 1 layer for medium). | - Prepare the grill with two zones: high and medium-low heat (for charcoal, this is 2 to 3 coals for the high zone and 1 layer for medium). |
| An explosion at a paraxylene plant in Fujian injured six and caused a fire. | An explosion at a paraxylene plant in Fujian injured five and caused a fire. |

<details><summary>写进提示词的卡片原文</summary>

```
keep a number from the source but make it count or measure a different thing.
Where it can be done: the sentence states a number that counts or measures one particular thing.
Why the result is an error: the number is in the source, but there it counts or measures something else.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: - Prepare the grill with two zones: high and medium-low heat (for charcoal, this is 2 to 3 coals thick for the high zone and 1 layer for medium).
  With the error: - Prepare the grill with two zones: high and medium-low heat (for charcoal, this is 2 to 3 coals for the high zone and 1 layer for medium).
- Faithful: An explosion at a paraxylene plant in Fujian injured six and caused a fire.
  With the error: An explosion at a paraxylene plant in Fujian injured five and caused a fire.
```

</details>

<a id="move_time_place"></a>

### 时间地点挂错（`move_time_place`）

| | |
|---|---|
| 做法 | keep a time or place from the source but tie it to a different event |
| 适用条件 | the sentence gives the time or place of an event, and the source describes another event |
| 为什么算错 | the time or place is in the source, but there it belongs to the other event |
| 怎么检查 | 在原文里找这个时间或地点，看它属于哪件事 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Attach the Monday Imax premiere to the film instead of the trailer footage.
- Attach the 12:21 a.m. time of the alarm company's call to police to the alarm's activation instead.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Based on the passages, you can transfer a GPX file to your Garmin device by connecting it to your computer with a USB cable, then copying or dragging the file into the NewFiles folder. | Based on the passages, you can transfer a GPX file to your Garmin device by connecting it to your computer with a USB cable, then copying or dragging the file into the NewFiles folder on your computer. |
| Kenyan police arrested five suspects following Thursday's attack by al-Shabaab militants at Garissa University College, which killed 147 people—including 142 students, three security officers, and two university security personnel—and injured 104, including 19 in critical condition. | On Thursday, Kenyan police arrested five suspects following an attack by al-Shabaab militants at Garissa University College, killing 147 people—including 142 students, three security officers, and two security personnel—and injuring 104, including 19 in critical condition. |

<details><summary>写进提示词的卡片原文</summary>

```
keep a time or place from the source but tie it to a different event.
Where it can be done: the sentence gives the time or place of an event, and the source describes another event.
Why the result is an error: the time or place is in the source, but there it belongs to the other event.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Based on the passages, you can transfer a GPX file to your Garmin device by connecting it to your computer with a USB cable, then copying or dragging the file into the NewFiles folder.
  With the error: Based on the passages, you can transfer a GPX file to your Garmin device by connecting it to your computer with a USB cable, then copying or dragging the file into the NewFiles folder on your computer.
- Faithful: Kenyan police arrested five suspects following Thursday's attack by al-Shabaab militants at Garissa University College, which killed 147 people—including 142 students, three security officers, and two university security personnel—and injured 104, including 19 in critical condition.
  With the error: On Thursday, Kenyan police arrested five suspects following an attack by al-Shabaab militants at Garissa University College, killing 147 people—including 142 students, three security officers, and two security personnel—and injuring 104, including 19 in critical condition.
```

</details>

<a id="merge_events"></a>

### 两处并成一句（`merge_events`）

| | |
|---|---|
| 做法 | take a detail the source gives about one event, rule or object and attach it to a different one described elsewhere in the source |
| 适用条件 | the source describes two different events, rules or objects, and the sentence is about one of them |
| 为什么算错 | every detail is in the source, but the source gives it about the other event, rule or object |
| 怎么检查 | 把句子拆成几个说法，分别回原文找，看它们说的是不是同一件事 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Attach the nominations, which came from Jag's HOH win, to his POV win.
- Attach the old rule's reset-at-Tribal-Council wrinkle to the current rule.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| 3. Press the Connect button at the back of the Xbox 360 controller. | 3. Press and hold the Connect button at the back of the Xbox 360 controller. |
| - Cook for 1 hour and 40 minutes. | - Cook for an additional 1 hour and 40 minutes. |

<details><summary>写进提示词的卡片原文</summary>

```
take a detail the source gives about one event, rule or object and attach it to a different one described elsewhere in the source.
Where it can be done: the source describes two different events, rules or objects, and the sentence is about one of them.
Why the result is an error: every detail is in the source, but the source gives it about the other event, rule or object.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: 3. Press the Connect button at the back of the Xbox 360 controller.
  With the error: 3. Press and hold the Connect button at the back of the Xbox 360 controller.
- Faithful: - Cook for 1 hour and 40 minutes.
  With the error: - Cook for an additional 1 hour and 40 minutes.
```

</details>

## 限定层

确定程度、范围、条件变了。

<a id="drop_hedge"></a>

### 删改推测词（`drop_hedge`）

| | |
|---|---|
| 做法 | remove or strengthen a hedge or an approximation so that a possibility, suspicion or estimate reads as settled or exact |
| 适用条件 | the sentence contains a hedge or an approximation such as 'may', 'likely', 'about', 'suspected' |
| 为什么算错 | the source only says it is possible, suspected or approximate |
| 怎么检查 | 在原文里找对应的说法，看有没有 may、likely、about、疑似这类词 |
| 出题前的规则 | 句子里有表示推测或约略的词 |

改法举例：

- Delete the hedge 'may be' so the claim becomes definite.
- Delete 'suspected to be', stating the tsunami origin as fact.
- Replace 'found no reason to doubt' with the stronger 'confirmed'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| A one-year-old white-and-black mix named Theia survived a near-death experience after being struck by a car, apparently hit on the head with a hammer, and buried in a field. | A one-year-old white-and-black mix named Theia survived a near-death experience after being struck by a car, hit on the head with a hammer, and buried in a field. |
| Early indications may include high fever, weakness, headache, sore throat, stomach pain, and diarrhea or constipation, with a rash on the trunk that typically resolves within 5 days. | Early indications may include high fever, weakness, headache, sore throat, stomach pain, and diarrhea or constipation, with a rash on the trunk that resolves within 5 days. |

<details><summary>写进提示词的卡片原文</summary>

```
remove or strengthen a hedge or an approximation so that a possibility, suspicion or estimate reads as settled or exact.
Where it can be done: the sentence contains a hedge or an approximation such as 'may', 'likely', 'about', 'suspected'.
Why the result is an error: the source only says it is possible, suspected or approximate.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: A one-year-old white-and-black mix named Theia survived a near-death experience after being struck by a car, apparently hit on the head with a hammer, and buried in a field.
  With the error: A one-year-old white-and-black mix named Theia survived a near-death experience after being struck by a car, hit on the head with a hammer, and buried in a field.
- Faithful: Early indications may include high fever, weakness, headache, sore throat, stomach pain, and diarrhea or constipation, with a rash on the trunk that typically resolves within 5 days.
  With the error: Early indications may include high fever, weakness, headache, sore throat, stomach pain, and diarrhea or constipation, with a rash on the trunk that resolves within 5 days.
```

</details>

<a id="add_hedge"></a>

### 加推测词（`add_hedge`）

| | |
|---|---|
| 做法 | add a hedge such as 'may', 'likely' or 'potentially' to something the source states as definite |
| 适用条件 | the sentence states something definite |
| 为什么算错 | the source states it without any doubt |
| 怎么检查 | 在原文里找对应的说法，看原文是不是说得肯定 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Add the hedge 'likely' to a cause of death the source states definitively.
- Add the hedge 'possible' to a fumigation the source treats as having happened.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| It is described as an avascular necrosis of the bone. | It is often described as an avascular necrosis of the bone. |
| Meanwhile, Finn and Hope bond over Sheila and Deacon's relationship and will plot to break them up. | Meanwhile, Finn and Hope bond over Sheila and Deacon's relationship but may plot to separate them. |

<details><summary>写进提示词的卡片原文</summary>

```
add a hedge such as 'may', 'likely' or 'potentially' to something the source states as definite.
Where it can be done: the sentence states something definite.
Why the result is an error: the source states it without any doubt.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: It is described as an avascular necrosis of the bone.
  With the error: It is often described as an avascular necrosis of the bone.
- Faithful: Meanwhile, Finn and Hope bond over Sheila and Deacon's relationship and will plot to break them up.
  With the error: Meanwhile, Finn and Hope bond over Sheila and Deacon's relationship but may plot to separate them.
```

</details>

<a id="drop_reported"></a>

### 删掉「据称」（`drop_reported`）

| | |
|---|---|
| 做法 | remove the marker that something is a report, allegation or one party's claim, so it reads as established fact |
| 适用条件 | the sentence marks something as a report, an allegation or one party's claim |
| 为什么算错 | the source only reports it as someone's claim, not as established fact |
| 怎么检查 | 在原文里看这件事是事实陈述，还是某一方的说法 |
| 出题前的规则 | 句子里有转述标记（说、据称、声称） |

改法举例：

- Delete 'allegedly' so the parking-without-blinker claim is stated as fact.
- Remove the attribution to Defense Minister Kenney, presenting his claim as fact.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| It is also noted as a symbol of wisdom in Buddhist iconography and is believed to stand guard at the entrance to the Gates of Paradise. | It is also noted as a symbol of wisdom in Buddhist iconography and stands at the entrance to the Gates of Paradise. |
| This event was part of China's "Strike Hard" campaign, which the government claims targets terrorism and separatism, and Amnesty International's annual review highlights a global trend of governments using the death penalty to combat crime and terrorism. | This event reflects China's "Strike Hard" campaign against terrorism and separatism, though Amnesty International's annual review highlights a global trend of governments using the death penalty to combat crime and terrorism. |

<details><summary>写进提示词的卡片原文</summary>

```
remove the marker that something is a report, allegation or one party's claim, so it reads as established fact.
Where it can be done: the sentence marks something as a report, an allegation or one party's claim.
Why the result is an error: the source only reports it as someone's claim, not as established fact.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: It is also noted as a symbol of wisdom in Buddhist iconography and is believed to stand guard at the entrance to the Gates of Paradise.
  With the error: It is also noted as a symbol of wisdom in Buddhist iconography and stands at the entrance to the Gates of Paradise.
- Faithful: This event was part of China's "Strike Hard" campaign, which the government claims targets terrorism and separatism, and Amnesty International's annual review highlights a global trend of governments using the death penalty to combat crime and terrorism.
  With the error: This event reflects China's "Strike Hard" campaign against terrorism and separatism, though Amnesty International's annual review highlights a global trend of governments using the death penalty to combat crime and terrorism.
```

</details>

<a id="plan_as_done"></a>

### 打算说成已发生（`plan_as_done`）

| | |
|---|---|
| 做法 | present a plan, aim, proposal, expectation or hypothetical as something that has happened or is certain to happen |
| 适用条件 | the sentence describes a plan, aim, proposal, expectation or hypothetical |
| 为什么算错 | in the source it has not happened: it is only planned, expected or proposed |
| 怎么检查 | 在原文里看这件事是已经发生，还是打算、预计、建议 |
| 出题前的规则 | 句子说的是计划、打算或预期 |

改法举例：

- Delete 'aims to' so touring in 2018 is stated as certain.
- Drop 'looks set to' so Germany is presented as having already joined.
- Change 'will be looking to win its first title since 1941' into 'secured its first title since 1941'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Denials from her boyfriend and friend, along with resistance to polygraph tests, deepened the mystery. | Denials from her ex-boyfriend and friend, along with resistance to polygraph tests, deepened the mystery. |
| The Department of Justice (DOJ) will soon begin negotiating to restructure the police department after a report revealed systemic racial discrimination, including illegal stops, excessive force, and a police department acting as a "collection agency" targeting African-American residents. | The Department of Justice (DOJ) is now negotiating to restructure the police department after a report revealed systemic racial discrimination, including illegal stops, excessive force, and a police department acting as a "collection agency" targeting African-American residents. |

<details><summary>写进提示词的卡片原文</summary>

```
present a plan, aim, proposal, expectation or hypothetical as something that has happened or is certain to happen.
Where it can be done: the sentence describes a plan, aim, proposal, expectation or hypothetical.
Why the result is an error: in the source it has not happened: it is only planned, expected or proposed.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Denials from her boyfriend and friend, along with resistance to polygraph tests, deepened the mystery.
  With the error: Denials from her ex-boyfriend and friend, along with resistance to polygraph tests, deepened the mystery.
- Faithful: The Department of Justice (DOJ) will soon begin negotiating to restructure the police department after a report revealed systemic racial discrimination, including illegal stops, excessive force, and a police department acting as a "collection agency" targeting African-American residents.
  With the error: The Department of Justice (DOJ) is now negotiating to restructure the police department after a report revealed systemic racial discrimination, including illegal stops, excessive force, and a police department acting as a "collection agency" targeting African-American residents.
```

</details>

<a id="generalize"></a>

### 范围放大（`generalize`）

| | |
|---|---|
| 做法 | widen the scope of a claim: drop a quantifier such as 'many', 'one of' or 'on average', or turn one named case into a group |
| 适用条件 | the sentence is about one named case or a part of a group, or has a quantifier such as 'many', 'one of', 'on average' |
| 为什么算错 | the source says it only of one case or of some, not of all |
| 怎么检查 | 在原文里看这句话说的是一个、一部分，还是全部 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Delete 'on average', so the $50+ increase applies to every retiree.
- Drop 'one of' and pluralize 'garrison' to 'garrisons'.
- Generalize one survivor's lament into survivors as a group.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| She has also been accused of considering running for office, prompting royal pundit PDina to urge her to focus on her duties as a duchess rather than politics. | She has also been accused of considering running for office, prompting royal pundits to urge her to focus on her duties as a duchess rather than politics. |
| Key evidence, including the flight data recorder, cockpit voice recorder, and DNA samples, has been collected. | All evidence, including the flight data recorder, cockpit voice recorder, and DNA samples, has been collected. |

<details><summary>写进提示词的卡片原文</summary>

```
widen the scope of a claim: drop a quantifier such as 'many', 'one of' or 'on average', or turn one named case into a group.
Where it can be done: the sentence is about one named case or a part of a group, or has a quantifier such as 'many', 'one of', 'on average'.
Why the result is an error: the source says it only of one case or of some, not of all.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: She has also been accused of considering running for office, prompting royal pundit PDina to urge her to focus on her duties as a duchess rather than politics.
  With the error: She has also been accused of considering running for office, prompting royal pundits to urge her to focus on her duties as a duchess rather than politics.
- Faithful: Key evidence, including the flight data recorder, cockpit voice recorder, and DNA samples, has been collected.
  With the error: All evidence, including the flight data recorder, cockpit voice recorder, and DNA samples, has been collected.
```

</details>

<a id="drop_restriction"></a>

### 删条件或限制（`drop_restriction`）

| | |
|---|---|
| 做法 | remove a condition or a restricting phrase so the claim covers more than the source says |
| 适用条件 | the sentence has a condition or a phrase that restricts what the claim covers |
| 为什么算错 | the source makes the claim only under that condition or within that range |
| 怎么检查 | 在原文里找这句话成立的条件和适用范围 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Drop the restriction 'from body fat' so that 'no fat loss' becomes 'no weight loss'.
- Delete the restriction 'who deals with infectious disease' so the county seems to have only one doctor overall.
- Delete the condition 'who entered the US by July 31'.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| A first draft of "The Breakfast Club" screenplay was recently found in a suburban Chicago school district filing cabinet. | A first draft of "The Breakfast Club" screenplay was recently found in a Chicago school district filing cabinet. |
| The levels of carbon monoxide and nicotine decline rapidly, and the heart and lungs begin to repair the damage caused by cigarette smoke. | The levels of carbon monoxide and nicotine decline rapidly, and the heart and lungs repair the damage caused by cigarette smoke. |

<details><summary>写进提示词的卡片原文</summary>

```
remove a condition or a restricting phrase so the claim covers more than the source says.
Where it can be done: the sentence has a condition or a phrase that restricts what the claim covers.
Why the result is an error: the source makes the claim only under that condition or within that range.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: A first draft of "The Breakfast Club" screenplay was recently found in a suburban Chicago school district filing cabinet.
  With the error: A first draft of "The Breakfast Club" screenplay was recently found in a Chicago school district filing cabinet.
- Faithful: The levels of carbon monoxide and nicotine decline rapidly, and the heart and lungs begin to repair the damage caused by cigarette smoke.
  With the error: The levels of carbon monoxide and nicotine decline rapidly, and the heart and lungs repair the damage caused by cigarette smoke.
```

</details>

## 句间层

句子之间的关系，或凭空加上的推论。

<a id="add_link"></a>

### 加或换连接词（`add_link`）

| | |
|---|---|
| 做法 | connect two statements from the source with a causal, contrastive or temporal connective the source does not use, or replace the source's connective with a different one |
| 适用条件 | the sentence, alone or with its neighbour, states two facts that the source does not connect as cause, contrast or sequence |
| 为什么算错 | the source states both facts but not that connection between them |
| 怎么检查 | 在原文里看这两件事之间有没有写明因果、转折或先后 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Link two separate source sentences with the causal connective 'as'.
- Change 'because he kept winning challenges' to 'despite winning challenges'.
- Prefix the sentence with the contrastive phrase 'Despite the incident', which the source does not contain.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| The event features free food, T-shirts, music, and prizes, with fans encouraged to arrive early and to create signs supporting the Buckeyes. | The event features free food, T-shirts, music, and prizes, with fans encouraged to arrive early to create signs supporting the Buckeyes. |
| Shreeves started her journey two years ago before moving to Ecuador, inspired by the No Trash Project. | Shreeves started her journey two years ago after moving to Ecuador, inspired by the No Trash Project. |

<details><summary>写进提示词的卡片原文</summary>

```
connect two statements from the source with a causal, contrastive or temporal connective the source does not use, or replace the source's connective with a different one.
Where it can be done: the sentence, alone or with its neighbour, states two facts that the source does not connect as cause, contrast or sequence.
Why the result is an error: the source states both facts but not that connection between them.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: The event features free food, T-shirts, music, and prizes, with fans encouraged to arrive early and to create signs supporting the Buckeyes.
  With the error: The event features free food, T-shirts, music, and prizes, with fans encouraged to arrive early to create signs supporting the Buckeyes.
- Faithful: Shreeves started her journey two years ago before moving to Ecuador, inspired by the No Trash Project.
  With the error: Shreeves started her journey two years ago after moving to Ecuador, inspired by the No Trash Project.
```

</details>

<a id="add_conclusion"></a>

### 加结论或评价（`add_conclusion`）

| | |
|---|---|
| 做法 | add an interpretation, cause, purpose or evaluation that the source does not state, or turn the source's silence on a point into an explicit statement |
| 适用条件 | any sentence that an interpretation, cause, purpose or evaluation could be attached to |
| 为什么算错 | the source does not state that interpretation, cause, purpose or evaluation |
| 怎么检查 | 在原文里找这个结论或评价，原文没说就是错 |
| 出题前的规则 | 无，任何句子都可以出这道题 |

改法举例：

- Turn 'issued a statement that did not reference pressurization issues' into 'stated no such problem existed'.
- Replace the unclear-direction correlation with a claim that marriage improves well-being.

真实例句：

| 忠于原文 | 写错的 |
|---|---|
| Dr. Mark Goodacre, a professor of New Testament studies at Duke University, discusses the Gospel of Mary Magdalene in response to viewer questions from CNN's "Finding Jesus." | Dr. Mark Goodacre, a professor of New Testament studies at Duke University, discusses the often-overlooked Gospel of Mary Magdalene in response to viewer questions from CNN's "Finding Jesus." |
| Child actress Suzanne Crough, who played Tracy on the 1970s sitcom "The Partridge Family," died at age 52 in Nevada. | Child actress Suzanne Crough, who played Tracy on the classic 1970s sitcom "The Partridge Family," died at age 52 in Nevada. |

<details><summary>写进提示词的卡片原文</summary>

```
add an interpretation, cause, purpose or evaluation that the source does not state, or turn the source's silence on a point into an explicit statement.
Where it can be done: any sentence that an interpretation, cause, purpose or evaluation could be attached to.
Why the result is an error: the source does not state that interpretation, cause, purpose or evaluation.
Real examples from other texts (only the edit differs between the two versions):
- Faithful: Dr. Mark Goodacre, a professor of New Testament studies at Duke University, discusses the Gospel of Mary Magdalene in response to viewer questions from CNN's "Finding Jesus."
  With the error: Dr. Mark Goodacre, a professor of New Testament studies at Duke University, discusses the often-overlooked Gospel of Mary Magdalene in response to viewer questions from CNN's "Finding Jesus."
- Faithful: Child actress Suzanne Crough, who played Tracy on the 1970s sitcom "The Partridge Family," died at age 52 in Nevada.
  With the error: Child actress Suzanne Crough, who played Tracy on the classic 1970s sitcom "The Partridge Family," died at age 52 in Nevada.
```

</details>
