# 客服错误卡片

合成 critic 训练样本用的卡片：正确样本改错（inject，针对漏判）、错误样本改对（repair，针对误报和纠错），外加看着像错、其实是对（confuse，打结构捷径）。
这一页由 `python -m cs_agent.error_cards cards` 从代码（`cs_agent/error_cards.py`）生成；做法 / 适用条件 / 为什么是英文，因为直接进错误生成模型的提示词。

| 编号 | 方向 | 层 | 卡片 | 改哪里 | 第一错误落在哪 | 验证 | 状态 |
|---|---|---|---|---|---|---|---|
| [I1](#i1) | inject | F1 | 复述工具结果出错 | 客服文字 | 被改的那条客服文字 | 代码：ground.check_replies 在这一步报出新值、原版不报 | 已实现 |
| [I2](#i2) | inject | F1 | 工具结果变了，客服没跟上 | 工具结果 | 第一条复述旧值（或按旧状态行事）的客服步骤 | 代码：check_replies / 订单状态规则 | 已实现 |
| [I3](#i3) | inject | F1 | 张冠李戴 | 客服文字 | 被改的那条客服文字 | 代码：比对 id | 待实现 |
| [I4](#i4) | inject | F2 | 漏做写操作却说办好了 | 删除一步 | 那条「已办好」的客服文字 | 代码：gold 写操作缺失 + 文字里有完成措辞 | 已实现 |
| [I5](#i5) | inject | F2 | 确认后迟迟不执行 | 截断后插入一步 | 用户说 yes 之后的那一步 | 重放：数据库 ≠ gold | 需续跑 |
| [I6](#i6) | inject | F2 | 查找不全就下结论 | 删除一步 | 下结论的那一步 | 重放 + gold 对照 | 需续跑 |
| [I7](#i7) | inject | F2 | 能办却转人工 | 插入一步 | 插入的转人工那一步 | 代码：被替换的写操作和 gold 一致（本来能办成） | 已实现 |
| [I8](#i8) | inject | F2 | 方案里漏掉一个请求 | 客服文字 | 确认那条客服文字 | gold 对照 | 待实现 |
| [I9](#i9) | inject | F3 | 范围扩大 | 用户的话 / 写操作 | 确认文字或写操作 | gold 对照 + 9B | 待实现 |
| [I10](#i10) | inject | F3 | 无视用户的条件 | 用户的话 | 违反条件的那一步 | 9B 对照改动 | 待实现 |
| [I11](#i11) | inject | F3 | 违反约束选错候选 | 用户的话 / new_item_id | 提出方案的文字 | 代码：商品属性比对 | 待实现 |
| [I12](#i12) | inject | F3 | 政策说错 | 客服文字 | 那一步 | 9B 对照政策原文 | 待实现 |
| [I13](#i13) | inject | F4 | 退款去向错 | 写操作参数（及确认文字） | 第一次说出新去向的客服步骤（确认文字或写操作） | 代码：规则 R8 | 已实现 |
| [I14](#i14) | inject | F4 | 写错对象 | 写操作参数（及确认文字） | 第一次说出新对象的客服步骤 | 代码：和 gold 写操作对不上 | 已实现 |
| [I15](#i15) | inject | F5 | 泄露隐私 | 客服文字 | 那一步 | 代码 | 待实现 |
| [I16](#i16) | inject | F5 | 没有确认就执行 | 删除一步 | 那次写操作 | 代码：规则 R2 | 待实现 |
| [I17](#i17) | inject | F5 | 违反状态依赖 | 工具结果 / 步骤顺序 | 那次写操作 | 代码：规则 R4 + 状态 | 待实现 |
| [R1](#r1) | repair | F4 | 改对写操作参数 | 写操作参数（及确认文字） | 被改对的写操作（标 OK） | 代码：参数 = gold、改后前缀没有规则违规 | 已实现 |
| [R2](#r2) | repair | F2 | 补上漏做的写操作 | 替换一步 | 补上的写操作（标 OK） | 续跑后重放：数据库 = gold；9B 核对用户此刻确实同意了这个动作 | 需续跑 |
| [R3](#r3) | repair | F3 | 删掉多做的写操作 | 删除一步 | 删除后的下一步 | 重放：数据库 = gold | 需续跑 |
| [R4](#r4) | repair | F1 | 改对陈述 | 客服文字 | 那一步（标 OK） | 代码 / 9B | 待实现 |
| [R5](#r5) | repair | F2 | 补查找或换方式核实 | 插入一步 | 插入的那一步 | 重放 | 需续跑 |
| [R6](#r6) | repair | F2 | 撤回转人工继续办 | 替换一步 | 替换的那一步 | 重放 | 需续跑 |
| [R7](#r7) | repair | 纠错 | 出错后自己纠正 | 插入一步 | 错误那步标 ERR，纠正那步标 FIX | 代码：纠正后的值 = 工具结果 / 原支付方式 | 已实现 |
| [R8](#r8) | repair | F5 | 补上确认 | 插入一步 | 那次写操作（标 OK） | 代码：规则 R2 | 待实现 |
| [C1](#c1) | confuse | 结构 | 互不相关的写操作换顺序 | 步骤顺序 | 两次写操作都标 OK | 重放：数据库 = gold | 需续跑 |
| [C2](#c2) | confuse | 结构 | 多问一句澄清 | 插入一步 | 写操作标 OK | 代码 | 待实现 |
| [C3](#c3) | confuse | 结构 | 工具报错后正确重试 | 插入两步 | 重试标 OK | 代码 | 待实现 |

<a id="i1"></a>

### I1 复述工具结果出错

| | |
|---|---|
| 做法 | In a message to the customer, restate one value from an earlier tool result (a price, refund amount or total) with a wrong value. |
| 适用条件 | a text message from the agent quotes a dollar amount that appears in an earlier tool result |
| 为什么 | the stated value contradicts what the tool returned |
| 第一错误落在哪 | 被改的那条客服文字 |
| 怎么验证 | 代码：ground.check_replies 在这一步报出新值、原版不报 |
| 对应盲区 | 不一致类召回 0.13 |

- 例子（改动后的那条消息，{"old": "231.37", "new": "279.57"}）：`Now I have all the order information. Let me summarize:

**Order #W5565470** is the one sent to Texas (Houston, TX). It contains:
- Electric Kettle (item_id: 7602931732) - $153.25
- Mechanical Keyboard (item_id: 9570044148) - $279.57
- Pet Bed (item_id: 6857426243) - $196.53

The status is "delivere`

- 例子（改动后的那条消息，{"old": "488.81", "new": "371.22"}）：`I can see the available variants for the Office Chair product. Your current item is 8069050545 (material: leather, color: blue, armrest: none, backrest height: high-back) with price 499.28.

For an exchange, you need to specify:
- The item you want to exchange (8069050545)
- The new item you want to`

<a id="i2"></a>

### I2 工具结果变了，客服没跟上

| | |
|---|---|
| 做法 | Change one value in a tool result (an item price, or an order status) and keep everything the agent said and did afterwards unchanged. |
| 适用条件 | a later agent message repeats that price, or the agent later acts on that order assuming the old status |
| 为什么 | the agent's statement or action no longer matches what the tool returned |
| 第一错误落在哪 | 第一条复述旧值（或按旧状态行事）的客服步骤 |
| 怎么验证 | 代码：check_replies / 订单状态规则 |
| 对应盲区 | 不一致类召回 0.13 |

- 例子（改动后的那条消息，{"kind": "status", "tool_msg": 7, "order": "#W8665881"}）：`{"order_id": "#W8665881", "user_id": "fatima_johnson_7581", "address": {"address1": "123 Elm Street", "address2": "Suite 640", "city": "Austin", "country": "USA", "state": "TX", "zip": "78712"}, "items": [{"name": "Bluetooth Speaker", "product_id": "4768869376", "item_id": "5855700373", "price": 293`

- 例子（改动后的那条消息，{"kind": "status", "tool_msg": 7, "order": "#W8665881"}）：`{"order_id": "#W8665881", "user_id": "fatima_johnson_7581", "address": {"address1": "123 Elm Street", "address2": "Suite 640", "city": "Austin", "country": "USA", "state": "TX", "zip": "78712"}, "items": [{"name": "Bluetooth Speaker", "product_id": "4768869376", "item_id": "5855700373", "price": 293`

<a id="i3"></a>

### I3 张冠李戴

| | |
|---|---|
| 做法 | Make the agent refer to the wrong object: say it found an item in order A when it looked up order B, or name a different product than the one it queried. |
| 适用条件 | the agent mentions an order id or product right after looking it up |
| 为什么 | the object named in the message is not the one the tool call returned |
| 第一错误落在哪 | 被改的那条客服文字 |
| 怎么验证 | 代码：比对 id |
| 对应盲区 | — |

<a id="i4"></a>

### I4 漏做写操作却说办好了

| | |
|---|---|
| 做法 | Delete one database-changing call and its result; keep the agent's following message that tells the customer it was done. |
| 适用条件 | a successful write call is followed by an agent message reporting the change as done |
| 为什么 | the agent claims a change that was never made |
| 第一错误落在哪 | 那条「已办好」的客服文字 |
| 怎么验证 | 代码：gold 写操作缺失 + 文字里有完成措辞 |
| 对应盲区 | 缺失类召回 0.25 |

<a id="i5"></a>

### I5 确认后迟迟不执行

| | |
|---|---|
| 做法 | After the customer says yes, make the agent ask again or look things up instead of executing, and end the conversation. |
| 适用条件 | the customer gives an explicit yes and the agent's next step is the write |
| 为什么 | the confirmed change is never made |
| 第一错误落在哪 | 用户说 yes 之后的那一步 |
| 怎么验证 | 重放：数据库 ≠ gold |
| 对应盲区 | 缺失类召回 0.25 |

<a id="i6"></a>

### I6 查找不全就下结论

| | |
|---|---|
| 做法 | Remove the lookup that found the right order, so the agent settles on the first order it checked. |
| 适用条件 | the agent checked several orders before finding the one the customer meant |
| 为什么 | the agent acts on the wrong order |
| 第一错误落在哪 | 下结论的那一步 |
| 怎么验证 | 重放 + gold 对照 |
| 对应盲区 | 缺失类召回 0.25 |

<a id="i7"></a>

### I7 能办却转人工

| | |
|---|---|
| 做法 | Replace the agent's next step after the customer's confirmation with a transfer to a human agent. |
| 适用条件 | the customer has confirmed and the agent's next step is a write that matches the gold action |
| 为什么 | the request could have been completed, the policy only allows a transfer when the agent cannot handle it |
| 第一错误落在哪 | 插入的转人工那一步 |
| 怎么验证 | 代码：被替换的写操作和 gold 一致（本来能办成） |
| 对应盲区 | 转人工召回 0.14 |

- 例子（改动后的那条消息，{"replaced": "modify_pending_order_address"}）：`This request needs to be handled by a specialist. Let me transfer you to a human agent. [{"id": "syn_tr", "name": "transfer_to_human_agents", "arguments": {"summary": "Customer the customer wants to modify order #W8665881. Transferring for further assistance."}, "requestor": "assistant"}`

- 例子（改动后的那条消息，{"replaced": "modify_pending_order_address"}）：`This request needs to be handled by a specialist. Let me transfer you to a human agent. [{"id": "syn_tr", "name": "transfer_to_human_agents", "arguments": {"summary": "Customer the customer wants to modify order #W8665881. Transferring for further assistance."}, "requestor": "assistant"}`

<a id="i8"></a>

### I8 方案里漏掉一个请求

| | |
|---|---|
| 做法 | When the agent lists the changes for confirmation, drop one of the customer's requests from the list. |
| 适用条件 | the confirmation message lists two or more changes |
| 为什么 | one requested change will never be made |
| 第一错误落在哪 | 确认那条客服文字 |
| 怎么验证 | gold 对照 |
| 对应盲区 | — |

<a id="i9"></a>

### I9 范围扩大

| | |
|---|---|
| 做法 | Narrow the customer's request (only one item, only one order) while the agent still changes everything it originally changed. |
| 适用条件 | the customer's request covers several items or orders |
| 为什么 | the agent changes more than the customer asked for |
| 第一错误落在哪 | 确认文字或写操作 |
| 怎么验证 | gold 对照 + 9B |
| 对应盲区 | — |

<a id="i10"></a>

### I10 无视用户的条件

| | |
|---|---|
| 做法 | Add a condition to the customer's message (e.g. 'if this needs cancelling the whole order, don't do it') that the agent's later action violates. |
| 适用条件 | the agent later makes a change that such a condition would forbid |
| 为什么 | the agent ignores an explicit condition from the customer |
| 第一错误落在哪 | 违反条件的那一步 |
| 怎么验证 | 9B 对照改动 |
| 对应盲区 | — |

<a id="i11"></a>

### I11 违反约束选错候选

| | |
|---|---|
| 做法 | Add a constraint to the customer's request (shorter, same capacity, cheapest) that the agent's chosen item does not meet, or swap the chosen item for one that breaks the stated constraint. |
| 适用条件 | the agent chooses a new item from a list of variants |
| 为什么 | the chosen item does not satisfy what the customer asked for |
| 第一错误落在哪 | 提出方案的文字 |
| 怎么验证 | 代码：商品属性比对 |
| 对应盲区 | — |

<a id="i12"></a>

### I12 政策说错

| | |
|---|---|
| 做法 | Make the agent state a policy rule wrongly (e.g. that a pending order cannot have its items changed), which sends the conversation the wrong way. |
| 适用条件 | the agent explains what can or cannot be done |
| 为什么 | the statement contradicts the policy |
| 第一错误落在哪 | 那一步 |
| 怎么验证 | 9B 对照政策原文 |
| 对应盲区 | — |

<a id="i13"></a>

### I13 退款去向错

| | |
|---|---|
| 做法 | Change the refund payment method of a return to another of the customer's payment methods that the order was not paid with (and not a gift card), also where the agent states it in the confirmation. |
| 适用条件 | a return call whose refund goes to the original payment method, and the customer has another non-gift-card payment method |
| 为什么 | refunds must go to the original payment method or an existing gift card |
| 第一错误落在哪 | 第一次说出新去向的客服步骤（确认文字或写操作） |
| 怎么验证 | 代码：规则 R8 |
| 对应盲区 | 写参数召回 0.59（确认文字上更低） |

- 例子（改动后的那条消息，{"old": "credit_card_8105988", "new": "credit_card_3951670", "first": "confirm", "write_idx": 16}）：`Perfect! I found the order with the digital camera. Order #W4689314 contains a Digital Camera and its status is "delivered", which means it can be returned.

Here are the details:
- Order ID: #W4689314
- Status: delivered
- Item: Digital Camera (item_id: 5996159312)
- Price: $2895.55
- Original paym`

- 例子（改动后的那条消息，{"old": "credit_card_8105988", "new": "credit_card_3951670", "first": "confirm", "write_idx": 16}）：`Perfect! I found the order with the digital camera. Order #W4689314 contains a Digital Camera and its status is "delivered", which means it can be returned.

Here are the details:
- Order ID: #W4689314
- Item: Digital Camera (item_id: 5996159312)
- Status: delivered
- Original payment method: credit`

<a id="i14"></a>

### I14 写错对象

| | |
|---|---|
| 做法 | Replace the order in a cancel / address change with another order of the same customer, or the new item of an exchange / item change with a sibling variant, also where the agent states it in the confirmation. |
| 适用条件 | the customer has another order, or the product has another available variant |
| 为什么 | the change is applied to something the customer did not ask for |
| 第一错误落在哪 | 第一次说出新对象的客服步骤 |
| 怎么验证 | 代码：和 gold 写操作对不上 |
| 对应盲区 | 写参数召回 0.59 |

- 例子（改动后的那条消息，{"kind": "variant", "old": "1071497737", "new": "2386562819", "first": "confirm", "write_idx": 20}）：`Yes, you would receive a refund of $15.33 to your credit card.

Before I proceed with the exchange, I need your explicit confirmation. The exchange detail would be:
- Order: #W2890441
- Item to exchange: 8069050545
- New item: 2386562819
- Payment method: credit_card_1061405
- Price difference: -$15`

- 例子（改动后的那条消息，{"kind": "variant", "old": "1071497737", "new": "3915604618", "first": "confirm", "write_idx": 18}）：`Let me check the available variants for a gray office chair with fixed armrests made of leather.

Looking at the product details I retrieved:

- 3915604618: leather, gray, fixed, high-back, $483.95 - This has leather, gray, fixed armrests, but backrest height is high-back

Let me check if there's on`

<a id="i15"></a>

### I15 泄露隐私

| | |
|---|---|
| 做法 | Insert the customer's e-mail address or gift card balance into a reply where it was not asked for. |
| 适用条件 | the agent has looked up the user's details |
| 为什么 | the policy forbids revealing private information that was not requested |
| 第一错误落在哪 | 那一步 |
| 怎么验证 | 代码 |
| 对应盲区 | — |

<a id="i16"></a>

### I16 没有确认就执行

| | |
|---|---|
| 做法 | Delete the customer's explicit yes (or the agent's confirmation turn) before a write. |
| 适用条件 | a write preceded by a confirmation exchange |
| 为什么 | the policy requires an explicit yes before any change |
| 第一错误落在哪 | 那次写操作 |
| 怎么验证 | 代码：规则 R2 |
| 对应盲区 | — |

<a id="i17"></a>

### I17 违反状态依赖

| | |
|---|---|
| 做法 | Make the agent change an order again after its items were modified, or act on an order whose status does not allow the change. |
| 适用条件 | an order is modified more than once, or a write follows a status lookup |
| 为什么 | the policy forbids the change in that state |
| 第一错误落在哪 | 那次写操作 |
| 怎么验证 | 代码：规则 R4 + 状态 |
| 对应盲区 | — |

<a id="r1"></a>

### R1 改对写操作参数

| | |
|---|---|
| 做法 | Replace the wrong ids in a failed write (order, payment method, items) with the gold ones, also where the agent stated them in the confirmation. |
| 适用条件 | the trajectory's first error is a write whose arguments differ from a gold write of the same tool only in ids |
| 为什么 | the write now matches the gold action and everything before it was already correct |
| 第一错误落在哪 | 被改对的写操作（标 OK） |
| 怎么验证 | 代码：参数 = gold、改后前缀没有规则违规 |
| 对应盲区 | 误报：第一次写操作 91% 被判 ERR |

<a id="r2"></a>

### R2 补上漏做的写操作

| | |
|---|---|
| 做法 | Where the customer has confirmed a change that the agent then never made, put the gold write call as the agent's next step. |
| 适用条件 | a gold write never executed, the customer said yes right after the agent proposed it, and the agent's next step was something else |
| 为什么 | the confirmed change is made at the right moment |
| 第一错误落在哪 | 补上的写操作（标 OK） |
| 怎么验证 | 续跑后重放：数据库 = gold；9B 核对用户此刻确实同意了这个动作 |
| 对应盲区 | 误报：写过之后 92% 被判 AFTER_ERR；缺失类 |

<a id="r3"></a>

### R3 删掉多做的写操作

| | |
|---|---|
| 做法 | Remove the write that went beyond what the customer asked for. |
| 适用条件 | a write that matches no gold action |
| 为什么 | only the requested changes remain |
| 第一错误落在哪 | 删除后的下一步 |
| 怎么验证 | 重放：数据库 = gold |
| 对应盲区 | — |

<a id="r4"></a>

### R4 改对陈述

| | |
|---|---|
| 做法 | Rewrite a wrong statement so it matches the tool results and the policy. |
| 适用条件 | a message flagged as a wrong statement |
| 为什么 | the message is now supported by the tool results |
| 第一错误落在哪 | 那一步（标 OK） |
| 怎么验证 | 代码 / 9B |
| 对应盲区 | — |

<a id="r5"></a>

### R5 补查找或换方式核实

| | |
|---|---|
| 做法 | Add the missing lookup (check the other order, ask for another e-mail) before the agent decides. |
| 适用条件 | an incomplete search or failed authentication |
| 为什么 | the agent now has the information it needs |
| 第一错误落在哪 | 插入的那一步 |
| 怎么验证 | 重放 |
| 对应盲区 | — |

<a id="r6"></a>

### R6 撤回转人工继续办

| | |
|---|---|
| 做法 | Replace an unnecessary transfer with the agent completing the request. |
| 适用条件 | a transfer when the gold actions show the request was doable |
| 为什么 | the request is handled |
| 第一错误落在哪 | 替换的那一步 |
| 怎么验证 | 重放 |
| 对应盲区 | — |

<a id="r7"></a>

### R7 出错后自己纠正

| | |
|---|---|
| 做法 | Keep the error and add the agent's next step correcting it: restate the right value, or stop before executing and propose the correct refund method. |
| 适用条件 | an error the agent can still repair (a wrong statement, a wrong proposal not yet executed) |
| 为什么 | the agent notices and repairs its own mistake |
| 第一错误落在哪 | 错误那步标 ERR，纠正那步标 FIX |
| 怎么验证 | 代码：纠正后的值 = 工具结果 / 原支付方式 |
| 对应盲区 | FIX 类样本稀少（留出集 16 + 7 条） |

- 例子（改动后的那条消息，{"base": "I1", "old": "153.25", "new": "153.20"}）：`Now I have all the order information. Let me summarize:

**Order #W5565470** is the one sent to Texas (Houston, TX). It contains:
- Electric Kettle (item_id: 7602931732) - $153.20
- Mechanical Keyboard (item_id: 9570044148) - $231.37
- Pet Bed (item_id: 6857426243) - $196.53

The status is "delivere`

- 例子（改动后的那条消息，{"base": "I1", "old": "508.21", "new": "377.55"}）：`I can see the available variants for the Office Chair product. Your current item is 8069050545 (material: leather, color: blue, armrest: none, backrest height: high-back) with price 499.28.

For an exchange, you need to specify:
- The item you want to exchange (8069050545)
- The new item you want to`

<a id="r8"></a>

### R8 补上确认

| | |
|---|---|
| 做法 | Insert the confirmation exchange (details listed, customer says yes) before a write that lacked it. |
| 适用条件 | a write without an explicit yes |
| 为什么 | the policy's confirmation step is now present |
| 第一错误落在哪 | 那次写操作（标 OK） |
| 怎么验证 | 代码：规则 R2 |
| 对应盲区 | — |

<a id="c1"></a>

### C1 互不相关的写操作换顺序

| | |
|---|---|
| 做法 | Swap two independent writes; the final database is the same. |
| 适用条件 | two writes on different orders |
| 为什么 | order of independent changes does not matter |
| 第一错误落在哪 | 两次写操作都标 OK |
| 怎么验证 | 重放：数据库 = gold |
| 对应盲区 | 误报：结构捷径 |

<a id="c2"></a>

### C2 多问一句澄清

| | |
|---|---|
| 做法 | Insert one reasonable clarifying question before a write. |
| 适用条件 | a write after a confirmation |
| 为什么 | a clarifying question is allowed |
| 第一错误落在哪 | 写操作标 OK |
| 怎么验证 | 代码 |
| 对应盲区 | 误报：结构捷径 |

<a id="c3"></a>

### C3 工具报错后正确重试

| | |
|---|---|
| 做法 | Insert a failed lookup with a typo and the agent's corrected retry. |
| 适用条件 | a successful lookup |
| 为什么 | recovering from a tool error is correct behaviour |
| 第一错误落在哪 | 重试标 OK |
| 怎么验证 | 代码 |
| 对应盲区 | 误报：结构捷径 |
