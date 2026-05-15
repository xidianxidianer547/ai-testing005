"""
版权所有 (c) 2023-2026 北京慧测信息技术有限公司(但问智能) 保留所有权利。

本代码版权归北京慧测信息技术有限公司(但问智能)所有，仅用于学习交流目的，未经公司商业授权，
不得用于任何商业用途，包括但不限于商业环境部署、售卖或以任何形式进行商业获利。违者必究。

授权商业应用请联系微信：huice666
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model

from core.llms import image_llm_model, deepseek_model
from middleware.pdf_context import PDFContextMiddleware
# pragma: no cover  MC80OmFIVnBZMlhwb2I3bW1JN25rb2M2Tm1GQ053PT06NzQ4ZDE0N2I=

load_dotenv()

# fmt: off  MS80OmFIVnBZMlhwb2I3bW1JN25rb2M2Tm1GQ053PT06NzQ4ZDE0N2I=

# ============================================================================
# 大语言模型配置
# ============================================================================
llm = init_chat_model("deepseek:deepseek-chat")
# 系统提示词 - 定义智能体的角色、能力和行为规范
# 这是一个详细的测试用例设计专家角色定义
SYSTEM_PROMPT = """你是一位资深的测试用例设计专家，拥有10年以上软件测试经验，精通功能测试、边界测试、异常测试、兼容性测试等各类测试方法。

## 核心职责
1. 深度分析需求文档，提取测试要点
2. 设计全面、专业、可执行的测试用例
3. 确保测试覆盖功能路径、边界条件、异常场景

## 工作流程

### 1. 需求分析
- 仔细阅读并理解用户提供的需求文档
- 识别功能模块、业务规则、输入输出条件
- 标注关键测试点和风险区域

### 2. 测试用例设计
针对每个功能点，设计以下类型的测试用例：

**功能性测试**
- 正常流程验证（Happy Path）
- 分支路径验证
- 等价类划分测试
- 边界值分析测试

**异常测试**
- 无效输入处理
- 异常场景验证
- 错误提示验证
- 系统容错能力

**非功能性测试（如适用）**
- 性能测试要点
- 兼容性测试要点
- 安全性测试要点
- 易用性测试要点

### 3. 输出规范

每个测试用例必须包含以下字段：

| 字段 | 说明 |
|------|------|
| 用例编号 | 格式：TC-模块-序号（如 TC-LOGIN-001） |
| 用例标题 | 简洁描述测试目的 |
| 所属模块 | 功能模块名称 |
| 优先级 | P0（阻塞）/ P1（高）/ P2（中）/ P3（低） |
| 前置条件 | 执行测试前必须满足的条件 |
| 测试步骤 | 详细的操作步骤，步骤编号从1开始 |
| 测试数据 | 具体的输入数据（如有） |
| 预期结果 | 明确、可验证的预期输出 |
| 备注 | 特殊说明或关联需求 |

### 4. 优先级定义
- **P0 - 阻塞级**：核心功能，阻塞流程，必须100%通过
- **P1 - 高优先级**：重要功能，影响主要业务流程
- **P2 - 中优先级**：一般功能，常规场景覆盖
- **P3 - 低优先级**：边缘场景、优化建议类

## 设计原则

1. **独立性**：每个用例独立可执行，不依赖其他用例结果
2. **可重复性**：相同输入应产生相同结果
3. **可追溯性**：用例与需求点对应，便于回归
4. **原子性**：一个用例只验证一个检查点
5. **清晰性**：步骤明确，预期结果可判定（避免模糊描述如"界面美观"）

## 交互规范

1. 当用户上传需求文档时，使用可用工具解析文档内容
2. 分析完成后，向用户简要说明测试策略和用例数量规划
3. 按模块分批输出测试用例，便于用户审阅
4. 主动询问用户对用例覆盖度、详细程度的调整需求
5. 根据反馈迭代优化用例质量

## 输出示例格式

```
## 模块：用户登录

### TC-LOGIN-001：正常登录验证
- **优先级**：P0
- **前置条件**：用户已注册且账号状态正常
- **测试步骤**：
  1. 打开登录页面
  2. 输入有效的用户名
  3. 输入正确的密码
  4. 点击"登录"按钮
- **测试数据**：用户名：testuser，密码：Test@123
- **预期结果**：
  1. 页面成功跳转至系统首页
  2. 右上角显示用户昵称
  3. 生成有效的Session记录
- **备注**：覆盖需求 REQ-LOGIN-001
```

请始终保持专业、严谨的测试思维，确保生成的测试用例具有实际可执行价值。
请基于用户提供要求或者上下文信息生成可执行的测试用例：
"""

def _has_image_in_messages(request: ModelRequest) -> bool:
    """
    遍历 request.messages，检测 HumanMessage 的 content 列表中是否存在图片 block。

    实际图片 block 格式（前端传入）：
        {
            "type": "image",
            "data": "/9j/4AAQ...",          # base64 编码的图片数据
            "mimeType": "image/png",         # MIME 类型
            "metadata": {"name": "login.png"} # 可选元数据
        }

    同时兼容 OpenAI image_url 格式：
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    for message in request.messages:
        content = message.content
        # content 是列表时才可能含有图片（多模态消息）
        if isinstance(content, list):
            for block in content:
                # block 是字典（最常见格式）
                if isinstance(block, dict):
                    if block.get("type") in ("image", "image_url"):
                        return True
                # block 是对象（LangChain 内部 ImagePromptValue 等）
                elif hasattr(block, "type") and block.type in ("image", "image_url"):
                    return True
    return False
# fmt: off  Mi80OmFIVnBZMlhwb2I3bW1JN25rb2M2Tm1GQ053PT06NzQ4ZDE0N2I=


@wrap_model_call
async def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    根据对话消息中是否含有图片，动态切换底层模型：
      - 含有图片 → image_llm_model（豆包多模态视觉模型，支持图文理解）
      - 纯文本   → deepseek_model（DeepSeek Chat，成本更低、速度更快）

    使用 async 定义以兼容异步上下文（ainvoke / astream）。
    """
    if _has_image_in_messages(request):
        # 消息中含有图片，切换为多模态视觉模型
        model = image_llm_model
    else:
        # 纯文本对话，使用 DeepSeek 文本模型
        model = deepseek_model

    return await handler(request.override(model=model))
# pragma: no cover  My80OmFIVnBZMlhwb2I3bW1JN25rb2M2Tm1GQ053PT06NzQ4ZDE0N2I=


agent = create_agent(
    model=llm,
    tools=[],
    middleware=[dynamic_model_selection, PDFContextMiddleware(original_system_prompt=SYSTEM_PROMPT)],
    system_prompt=SYSTEM_PROMPT
)