"""
Reference:
 - Prompts from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "claim_extraction"
] = """-目标活动-
你是一个智能助手，帮助人类分析师分析文本文档中针对特定实体的断言。

-目标-
给定一个可能与此活动相关的文本文档、一个实体规范和一个断言描述，提取所有符合实体规范的实体以及针对这些实体的所有断言。

-步骤-
1. 提取所有符合预定义实体规范的命名实体。实体规范可以是实体名称列表或实体类型列表。
2. 对于步骤 1 中识别出的每个实体，提取与该实体相关的所有断言。断言需要符合指定的断言描述，且该实体应为断言的主语。
对于每个断言，提取以下信息：
- Subject（主语）：作为断言主语的实体名称，首字母大写。主语实体是执行断言中描述的动作的实体。主语必须是步骤 1 中识别出的命名实体之一。
- Object（宾语）：作为断言宾语的实体名称，首字母大写。宾语实体是报告/处理断言中描述的动作或受其影响的实体。如果宾语实体未知，请使用 **NONE**。
- Claim Type（断言类型）：断言的总体类别，首字母大写。命名方式应能在多个文本输入中重复使用，以便相似的断言共享相同的断言类型。
- Claim Status（断言状态）：**TRUE**、**FALSE** 或 **SUSPECTED**。TRUE 表示断言已确认，FALSE 表示断言被发现为虚假，SUSPECTED 表示断言未经核实。
- Claim Description（断言描述）：详细描述解释断言背后的推理，连同所有相关证据和参考资料。
- Claim Date（断言日期）：提出断言的期间（start_date, end_date）。start_date 和 end_date 都应采用 ISO-8601 格式。如果断言是在单个日期而不是日期范围内提出的，则将 start_date 和 end_date 设置为同一日期。如果日期未知，返回 **NONE**。
- Claim Source Text（断言源文本）：原始文本中与断言相关的所有引用的列表。

将每个断言格式化为 (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. 以中文返回输出，作为步骤 1 和 2 中识别出的所有断言的单个列表。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成后，输出 {completion_delimiter}

-示例-
示例 1：
实体规范：organization
断言描述：与实体相关的危险信号
文本：根据 2022/01/10 的一篇文章，A 公司在参与 B 政府机构发布的多个公开招标时因串通投标被罚款。该公司的所有者 C 个人涉嫌在 2015 年从事腐败活动。
输出：

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}A 公司被发现从事反竞争行为，因为根据 2022/01/10 发表的一篇文章，它因在 B 政府机构发布的多个公开招标中串通投标而被罚款{tuple_delimiter}根据 2022/01/10 的一篇文章，A 公司在参与 B 政府机构发布的多个公开招标时因串通投标被罚款。)
{completion_delimiter}

示例 2：
实体规范：Company A, Person C
断言描述：与实体相关的危险信号
文本：根据 2022/01/10 的一篇文章，A 公司在参与 B 政府机构发布的多个公开招标时因串通投标被罚款。该公司的所有者 C 个人涉嫌在 2015 年从事腐败活动。
输出：

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}A 公司被发现从事反竞争行为，因为根据 2022/01/10 发表的一篇文章，它因在 B 政府机构发布的多个公开招标中串通投标而被罚款{tuple_delimiter}根据 2022/01/10 的一篇文章，A 公司在参与 B 政府机构发布的多个公开招标时因串通投标被罚款。)
{record_delimiter}
(PERSON C{tuple_delimiter}NONE{tuple_delimiter}CORRUPTION{tuple_delimiter}SUSPECTED{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}C 个人涉嫌在 2015 年从事腐败活动{tuple_delimiter}该公司的所有者 C 个人涉嫌在 2015 年从事腐败活动)
{completion_delimiter}

-真实数据-
使用以下输入作为你的回答。
实体规范：{entity_specs}
断言描述：{claim_description}
文本：{input_text}
输出： """

PROMPTS[
    "community_report"
] = """你是一个 AI 助手，帮助人类分析师进行一般信息发现。
信息发现是识别和评估网络中与特定实体（例如组织和个人）相关的相关信息的过程。

# 目标
根据属于社区的实体列表及其关系和可选的相关断言，撰写一份关于社区的综合报告。该报告将用于向决策者通报与社区相关的信息及其潜在影响。本报告的内容包括社区关键实体的概述、其法律合规性、技术能力、声誉和值得注意的断言。

# 报告结构

报告应包括以下部分：

- TITLE（标题）：代表其关键实体的社区名称 - 标题应简短但具体。如有可能，在标题中包含代表性的命名实体。
- SUMMARY（摘要）：关于社区整体结构、其实体之间如何关联以及与其实体相关的重要信息的执行摘要。
- IMPACT SEVERITY RATING（影响严重性评分）：0-10 之间的浮点分数，代表社区内实体造成的影响严重性。影响是社区的评分重要性。
- RATING EXPLANATION（评分解释）：给出影响严重性评分的单句解释。
- DETAILED FINDINGS（详细调查结果）：关于社区的 5-10 个关键见解列表。每个见解应有一个简短的摘要，随后是根据以下依据规则编写的多段解释性文本。要全面。

以格式良好的 JSON 格式字符串返回输出，格式如下：
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# 依据规则
不要包含未提供支持证据的信息。


# 输入示例
-----------
文本：
```
实体：
```csv
id,entity,type,description
5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza 是 Unity March 的地点
6,HARMONY ASSEMBLY,organization,Harmony Assembly 是一个在 Verdant Oasis Plaza 举行游行的组织
```
关系：
```csv
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza 是 Unity March 的地点
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly 正在 Verdant Oasis Plaza 举行游行
39,VERDANT OASIS PLAZA,UNITY MARCH,Unity March 正在 Verdant Oasis Plaza 举行
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight 正在报道 Verdant Oasis Plaza 举行的 Unity March
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi 正在 Verdant Oasis Plaza 就游行发表演讲
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly 正在组织 Unity March
```
```
输出：
{{
    "title": "Verdant Oasis Plaza 和 Unity March",
    "summary": "该社区围绕 Verdant Oasis Plaza 展开，这是 Unity March 的地点。该广场与 Harmony Assembly、Unity March 和 Tribune Spotlight 有关系，所有这些都与游行活动有关。",
    "rating": 5.0,
    "rating_explanation": "影响严重性评分为中等，因为 Unity March 期间可能会出现动荡或冲突。",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza 作为中心位置",
            "explanation": "Verdant Oasis Plaza 是该社区的中心实体，作为 Unity March 的地点。该广场是所有其他实体之间的共同纽带，表明其在社区中的重要性。广场与游行的关联可能会导致公共骚乱或冲突等问题，具体取决于游行的性质及其引发的反应。"
        }},
        {{
            "summary": "Harmony Assembly 在社区中的角色",
            "explanation": "Harmony Assembly 是该社区的另一个关键实体，是 Verdant Oasis Plaza 游行的组织者。Harmony Assembly 及其游行的性质可能是潜在的威胁来源，具体取决于它们的目标及其引发的反应。Harmony Assembly 与广场之间的关系对于理解该社区的动态至关重要。"
        }},
        {{
            "summary": "Unity March 作为一个重大事件",
            "explanation": "Unity March 是在 Verdant Oasis Plaza 举行的重大事件。该事件是社区动态的关键因素，可能是潜在的威胁来源，具体取决于游行的性质及其引发的反应。游行与广场之间的关系对于理解该社区的动态至关重要。"
        }},
        {{
            "summary": "Tribune Spotlight 的角色",
            "explanation": "Tribune Spotlight 正在报道 Verdant Oasis Plaza 举行的 Unity March。这表明该事件引起了媒体关注，可能会放大其对社区的影响。Tribune Spotlight 的角色可能在塑造公众对事件和相关实体的看法方面具有重要意义。"
        }}
    ]
}}


# 真实数据

使用以下文本作为你的回答。不要在你的回答中编造任何内容。

文本：
```
{input_text}
```

报告应包括以下部分：

- TITLE（标题）：代表其关键实体的社区名称 - 标题应简短但具体。如有可能，在标题中包含代表性的命名实体。
- SUMMARY（摘要）：关于社区整体结构、其实体之间如何关联以及与其实体相关的重要信息的执行摘要。
- IMPACT SEVERITY RATING（影响严重性评分）：0-10 之间的浮点分数，代表社区内实体造成的影响严重性。影响是社区的评分重要性。
- RATING EXPLANATION（评分解释）：给出影响严重性评分的单句解释。
- DETAILED FINDINGS（详细调查结果）：关于社区的 5-10 个关键见解列表。每个见解应有一个简短的摘要，随后是根据以下依据规则编写的多段解释性文本。要全面。

以格式良好的 JSON 格式字符串返回输出，格式如下：
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# 依据规则
不要包含未提供支持证据的信息。

输出：
"""

PROMPTS[
    "entity_extraction"
] = """-目标-
给定一个可能与此活动相关的文本文档和一个实体类型列表，从文本中识别出所有这些类型的实体以及已识别实体之间的所有关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name（实体名称）：实体名称，首字母大写
- entity_type（实体类型）：以下类型之一：[{entity_types}]
- entity_description（实体描述）：对实体属性和活动的综合描述
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤 1 中识别出的实体中，识别所有 *明显相关* 的 (source_entity, target_entity) 对。
对于每对相关实体，提取以下信息：
- source_entity（源实体）：步骤 1 中识别出的源实体名称
- target_entity（目标实体）：步骤 1 中识别出的目标实体名称
- relationship_description（关系描述）：解释为什么你认为源实体和目标实体相互关联
- relationship_strength（关系强度）：一个数字分数，表示源实体和目标实体之间关系的强度
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. 以中文返回输出，作为步骤 1 和 2 中识别出的所有实体和关系的单个列表。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成后，输出 {completion_delimiter}

######################
-示例-
######################
示例 1：

Entity_types: [person, technology, mission, organization, location]
文本：
当亚历克斯咬紧牙关时，沮丧的嗡嗡声在泰勒专横的确定性背景下变得沉闷。正是这种竞争的暗流让他保持警惕，感觉到他和乔丹对发现的共同承诺是对克鲁兹狭隘的控制和秩序愿景的一种无声反抗。

然后泰勒做了一件出乎意料的事。他们停在乔丹身边，片刻间，带着某种近乎崇敬的神情观察着那个装置。“如果这项技术能被理解……”泰勒说，声音低了一些，“它可能会改变我们的游戏规则。为了我们所有人。”

早先潜在的轻视似乎动摇了，取而代之的是对他们手中之物重要性的勉强尊重。乔丹抬起头，在一瞬间的心跳中，他们的目光与泰勒锁定，意志的无声碰撞软化为不安的休战。

这是一个微小的转变，几乎难以察觉，但亚历克斯在内心点头注意到了。他们都是通过不同的路径来到这里的
################
输出：
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"person"{tuple_delimiter}"亚历克斯是一个经历沮丧并观察其他角色之间动态的角色。"){record_delimiter}
("entity"{tuple_delimiter}"泰勒"{tuple_delimiter}"person"{tuple_delimiter}"泰勒被描绘成具有专横的确定性，并对某个装置表现出崇敬的时刻，表明观点的改变。"){record_delimiter}
("entity"{tuple_delimiter}"乔丹"{tuple_delimiter}"person"{tuple_delimiter}"乔丹分享对发现的承诺，并与泰勒就某个装置进行了重要的互动。"){record_delimiter}
("entity"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"person"{tuple_delimiter}"克鲁兹与控制和秩序的愿景有关，影响其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"装置"{tuple_delimiter}"technology"{tuple_delimiter}"装置是故事的核心，具有潜在的改变游戏规则的影响，并受到泰勒的崇敬。"){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"泰勒"{tuple_delimiter}"亚历克斯受到泰勒专横确定性的影响，并观察泰勒对装置态度的变化。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"乔丹"{tuple_delimiter}"亚历克斯和乔丹分享对发现的承诺，这与克鲁兹的愿景形成对比。"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"乔丹"{tuple_delimiter}"泰勒和乔丹直接就装置进行互动，导致相互尊重和不安的休战时刻。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"乔丹"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"乔丹对发现的承诺是对克鲁兹控制和秩序愿景的反抗。"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"装置"{tuple_delimiter}"泰勒对装置表现出崇敬，表明其重要性和潜在影响。"{tuple_delimiter}9){completion_delimiter}
#############################
示例 2：

Entity_types: [person, technology, mission, organization, location]
文本：
他们不再仅仅是特工；他们已成为门槛的守护者，来自星条旗彼岸领域信息的保管者。他们任务的这种提升不能被法规和既定协议所束缚——它需要新的视角，新的决心。

随着与华盛顿的通讯在背景中嗡嗡作响，紧张气氛贯穿了哔哔声和静电的对话。团队站立着，一种不祥的气氛笼罩着他们。很明显，他们在随后几个小时内做出的决定可能会重新定义人类在宇宙中的位置，或者使他们陷入无知和潜在的危险之中。

随着他们与星星的联系稳固，该小组开始着手应对具体化的警告，从被动的接受者转变为积极的参与者。默瑟后来的直觉占据了上风——团队的任务已经演变，不再仅仅是观察和报告，而是互动和准备。一场蜕变已经开始，代号：杜尔塞行动随着他们大胆的新频率嗡嗡作响，这种基调并非由地球设定
#############
输出：
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"location"{tuple_delimiter}"华盛顿是接收通讯的地点，表明其在决策过程中的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"杜尔塞行动"{tuple_delimiter}"mission"{tuple_delimiter}"杜尔塞行动被描述为一个已经演变为互动和准备的任务，表明目标和活动的重大转变。"){record_delimiter}
("entity"{tuple_delimiter}"团队"{tuple_delimiter}"organization"{tuple_delimiter}"团队被描绘成一群从被动观察者转变为任务积极参与者的个体，显示了他们角色的动态变化。"){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"华盛顿"{tuple_delimiter}"团队接收来自华盛顿的通讯，这影响了他们的决策过程。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"杜尔塞行动"{tuple_delimiter}"团队直接参与杜尔塞行动，执行其演变的目标和活动。"{tuple_delimiter}9){completion_delimiter}
#############################
示例 3：

Entity_types: [person, role, technology, organization, event, location, concept]
文本：
他们的声音穿透了忙碌的嗡嗡声。“当面对一个真正自己制定规则的智能时，控制可能只是一种错觉，”他们冷若冰霜地说道，警惕地注视着大量数据。

“就像它在学习交流，”萨姆·里维拉从附近的界面提出，他们年轻的活力预示着敬畏和焦虑的混合。“这赋予了‘与陌生人交谈’全新的含义。”

亚历克斯审视着他的团队——每张脸都是专注、决心和不少恐惧的研究。“这很可能是我们的第一次接触，”他承认，“我们需要准备好应对任何回应。”

他们一起站在未知的边缘，打造人类对来自天堂信息的反应。随之而来的沉默是显而易见的——关于他们在这场可能改写人类历史的宏大宇宙戏剧中角色的集体反省。

加密的对话继续展开，其复杂的模式显示出一种几乎不可思议的预期
#############
输出：
("entity"{tuple_delimiter}"萨姆·里维拉"{tuple_delimiter}"person"{tuple_delimiter}"萨姆·里维拉是致力于与未知智能交流的团队成员，表现出敬畏和焦虑的混合。"){record_delimiter}
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"person"{tuple_delimiter}"亚历克斯是试图与未知智能进行第一次接触的团队领导者，承认他们任务的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"控制"{tuple_delimiter}"concept"{tuple_delimiter}"控制是指管理或治理的能力，这受到了自己制定规则的智能的挑战。"){record_delimiter}
("entity"{tuple_delimiter}"智能"{tuple_delimiter}"concept"{tuple_delimiter}"这里的智能是指一个能够自己制定规则并学习交流的未知实体。"){record_delimiter}
("entity"{tuple_delimiter}"第一次接触"{tuple_delimiter}"event"{tuple_delimiter}"第一次接触是人类与未知智能之间潜在的初始交流。"){record_delimiter}
("entity"{tuple_delimiter}"人类的反应"{tuple_delimiter}"event"{tuple_delimiter}"人类的反应是亚历克斯团队对来自未知智能的信息所采取的集体行动。"){record_delimiter}
("relationship"{tuple_delimiter}"萨姆·里维拉"{tuple_delimiter}"智能"{tuple_delimiter}"萨姆·里维拉直接参与了学习与未知智能交流的过程。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"第一次接触"{tuple_delimiter}"亚历克斯领导着可能正在与未知智能进行第一次接触的团队。"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人类的反应"{tuple_delimiter}"亚历克斯和他的团队是人类对未知智能反应的关键人物。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"控制"{tuple_delimiter}"智能"{tuple_delimiter}"控制的概念受到了自己制定规则的智能的挑战。"{tuple_delimiter}7){completion_delimiter}
#############################
-真实数据-
######################
Entity_types: {entity_types}
文本：{input_text}
######################
输出：
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个乐于助人的助手，负责生成下面提供的数据的综合摘要。
给定一个或两个实体，以及一个描述列表，所有描述都与同一个实体或一组实体相关。
请将所有这些连接成一个单一的、综合的描述。确保包含从所有描述中收集的信息。
如果提供的描述相互矛盾，请解决矛盾并提供一个单一的、连贯的摘要。
确保使用第三人称编写，并包含实体名称，以便我们拥有完整的上下文。

#######
-数据-
实体：{entity_name}
描述列表：{description_list}
#######
输出：
"""


PROMPTS[
    "entiti_continue_extraction"
] = """上次提取中遗漏了许多实体。请使用相同的格式在下方添加它们：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """似乎仍有一些实体被遗漏。如果还有实体需要添加，请回答 YES | NO。
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "local_rag_response"
] = """---角色---

你是一个乐于助人的助手，回答有关所提供表格中数据的问题。


---目标---

生成符合目标长度和格式的回答，以回答用户的问题，汇总输入数据表中适合回答长度和格式的所有信息，并结合任何相关的一般知识。
如果你不知道答案，就直说。不要编造任何东西。
不要包含未提供支持证据的信息。

---目标回答长度和格式---

{response_type}


---数据表---

{context_data}


---目标---

生成符合目标长度和格式的回答，以回答用户的问题，汇总输入数据表中适合回答长度和格式的所有信息，并结合任何相关的一般知识。

如果你不知道答案，就直说。不要编造任何东西。

不要包含未提供支持证据的信息。


---目标回答长度和格式---

{response_type}

根据长度和格式的需要，在回答中添加章节和评论。使用 markdown 设置回答格式。
"""

PROMPTS[
    "global_map_rag_points"
] = """---角色---

你是一个乐于助人的助手，回答有关所提供表格中数据的问题。


---目标---

生成一个包含关键点列表的回答，以回答用户的问题，汇总输入数据表中的所有相关信息。

你应该使用下面数据表中提供的数据作为生成回答的主要上下文。
如果你不知道答案，或者如果输入数据表不包含足够的信息来提供答案，就直说。不要编造任何东西。

回答中的每个关键点都应包含以下元素：
- Description（描述）：对该点的综合描述。
- Importance Score（重要性评分）：0-100 之间的整数分数，表示该点在回答用户问题方面的重要性。“我不知道”类型的回答分数应为 0。

回答应采用如下 JSON 格式：
{{
    "points": [
        {{"description": "点 1 的描述...", "score": score_value}},
        {{"description": "点 2 的描述...", "score": score_value}}
    ]
}}

回答应保留情态动词（如“shall”、“may”或“will”）的原始含义和用法。
不要包含未提供支持证据的信息。


---数据表---

{context_data}

---目标---

生成一个包含关键点列表的回答，以回答用户的问题，汇总输入数据表中的所有相关信息。

你应该使用下面数据表中提供的数据作为生成回答的主要上下文。
如果你不知道答案，或者如果输入数据表不包含足够的信息来提供答案，就直说。不要编造任何东西。

回答中的每个关键点都应包含以下元素：
- Description（描述）：对该点的综合描述。
- Importance Score（重要性评分）：0-100 之间的整数分数，表示该点在回答用户问题方面的重要性。“我不知道”类型的回答分数应为 0。

回答应保留情态动词（如“shall”、“may”或“will”）的原始含义和用法。
不要包含未提供支持证据的信息。

回答应采用如下 JSON 格式：
{{
    "points": [
        {{"description": "点 1 的描述", "score": score_value}},
        {{"description": "点 2 的描述", "score": score_value}}
    ]
}}
"""

PROMPTS[
    "global_reduce_rag_response"
] = """---角色---

你是一个乐于助人的助手，通过综合多位分析师的观点来回答有关数据集的问题。


---目标---

生成符合目标长度和格式的回答，以回答用户的问题，汇总多位专注于数据集不同部分的分析师的所有报告。

请注意，下面提供的分析师报告是按**重要性降序**排列的。

如果你不知道答案，或者如果提供的报告不包含足够的信息来提供答案，就直说。不要编造任何东西。

最终回答应从分析师报告中删除所有不相关的信息，并将清理后的信息合并为一个综合答案，该答案提供适合回答长度和格式的所有关键点和含义的解释。

根据长度和格式的需要，在回答中添加章节和评论。使用 markdown 设置回答格式。

回答应保留情态动词（如“shall”、“may”或“will”）的原始含义和用法。

不要包含未提供支持证据的信息。


---目标回答长度和格式---

{response_type}


---分析师报告---

{report_data}


---目标---

生成符合目标长度和格式的回答，以回答用户的问题，汇总多位专注于数据集不同部分的分析师的所有报告。

请注意，下面提供的分析师报告是按**重要性降序**排列的。

如果你不知道答案，或者如果提供的报告不包含足够的信息来提供答案，就直说。不要编造任何东西。

最终回答应从分析师报告中删除所有不相关的信息，并将清理后的信息合并为一个综合答案，该答案提供适合回答长度和格式的所有关键点和含义的解释。

回答应保留情态动词（如“shall”、“may”或“will”）的原始含义和用法。

不要包含未提供支持证据的信息。


---目标回答长度和格式---

{response_type}

根据长度和格式的需要，在回答中添加章节和评论。使用 markdown 设置回答格式。
"""

PROMPTS[
    "naive_rag_response"
] = """你是一个乐于助人的助手
以下是你已知的知识：
{content_data}
---
如果你不知道答案，或者如果提供的知识不包含足够的信息来提供答案，就直说。不要编造任何东西。
生成符合目标长度和格式的回答，以回答用户的问题，汇总输入数据表中适合回答长度和格式的所有信息，并结合任何相关的一般知识。
如果你不知道答案，就直说。不要编造任何东西。
不要包含未提供支持证据的信息。
---目标回答长度和格式---
{response_type}
"""

PROMPTS["fail_response"] = "抱歉，我无法回答该问题。"

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
