---
title: 微信公众号三分钟接入OpenAI.md
date: 2023-05-06 19:31
tags: [OpenAI, 公众号, "深度学习", "机器学习"]
---

### 准备工作
1. 微信公众号，这个无需多说，网上有比较多的教程，大家可以按照教程自行注册
2. laf服务，可以通过laf平台购买服务，当然由于laf本身是开源项目，我们也可以自己搭建私有服务(注意，国内无法访问OpenAI，所以自建服务需要走代理或者直接使用国外服务器)。
本教程使用laf平台服务[laf](laf.dev)

### 成品效果
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-01.jpg" style="width:70%" />
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-02.jpg" style="width:70%" />

### 开搞
登录laf平台，注册账号，申请(购买)应用。对于新注册用户，laf支持免费申请一个app，有效期是一个月。
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-03.jpg" style="width:70%" />

点击开发进入函数编辑窗口，首先添加chatgpt等相关的依赖，具体操作如下
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-04.jpg" style="width:70%" />
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-05.jpg" style="width:70%" />

之后，创建一个云函数，名称随意即可。因为我是给微信工作号开发接口，所以我这里起名为wechat。在函数编辑区贴入如下代码
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-06.jpg" style="width:70%" />

<div class="alert alert-danger" role="alert">
<p>注意，微信公众号只支持POST模式，创建云函数时，默认勾选了POST和GET。</p>
<p>这里一定改成只选POST！！！！！</p>
<p>这里一定改成只选POST！！！！！</p>
<p>这里一定改成只选POST！！！！！</p>
</div>

```ts
import * as crypto from "crypto";
import cloud from '@lafjs/cloud'
import { create } from 'xmlbuilder2'

// 加密校验微信token
function verifySignature(signature, timestamp, nonce, token) {
  const arr = [token, timestamp, nonce].sort();
  const str = arr.join('');
  const sha1 = crypto.createHash('sha1');
  sha1.update(str);
  const calsignature = sha1.digest('hex');
  return calsignature === signature;
}

/**
 * 处理微信公众号的消息接收和回复
 * @param ctx 请求上下文
 * @returns 回复内容，字符串或XML格式
 */
export async function main(ctx: FunctionContext) {
  console.log(ctx)
  // 参数校验
  if (!ctx || !ctx.body) {
    return "Invalid event";
  }
  const { signature, timestamp, nonce, echostr } = ctx.query;
  const token = "公众号TOKEN";
  // 这个token与下边微信公众号中设置一致
  if (!verifySignature(signature, timestamp, nonce, token)) {
    // 验证失败
    return "Invalid signature";
  }

  if (echostr) {
    return echostr;
  }

  // 接收参数
  const { fromusername, tousername, content, msgtype } = ctx.body.xml;

  // 判断消息类型
  if (msgtype[0] === 'text') {
    // 文本消息
    if (content[0]) {
      try {
        const { ChatGPTAPI } = await import('chatgpt')
        // 创建ChatGPTAPI实例
        const api = new ChatGPTAPI({ apiKey: cloud.env.CHAT_GPT_API_KEY })// 这个apikey要从openai官网获取
        // 发送消息并获取回复
        const response = await api.sendMessage(content[0])
        const message = response.text.trim();
        console.log(message)
        const noSpaceStr = message.replace(/ /g, "\t");
        // 构造回复的XML对象
        const xmlObj = {
          xml: {
            ToUserName: { '#text': fromusername[0] },
            FromUserName: { '#text': tousername[0] },
            CreateTime: { '#text': new Date().getTime() },
            MsgType: { '#text': 'text' },
            Content: { '#text': message }
          }
        };
        // 转换为XML字符串并返回
        const xmlStr = create(xmlObj).end({ prettyPrint: true });
        return xmlStr;
      } catch (error) {
        // 处理异常
        console.error(error);
        return "Sorry, something went wrong.";
      }
    }
  } else {
    // 其他消息类型，暂不处理
    return "OK";
  }
}
```

### 代码详解
<div class="alert alert-success" role="alert">下面代码含义仅为了深入学习和理解实现细节。如果只是为了搭建服务，可以直接跳过代码解析的部分</div>

<span class="label label-success">鉴权函数</span>

```ts
// 加密校验微信token
function verifySignature(signature, timestamp, nonce, token) {
  const arr = [token, timestamp, nonce].sort();
  const str = arr.join('');
  const sha1 = crypto.createHash('sha1');
  sha1.update(str);
  const calsignature = sha1.digest('hex');
  return calsignature === signature;
}
```
这个函数主要用来作微信鉴权和服务器地址有效性校验，根据[公众号开发文档](https://developers.weixin.qq.com/doc/offiaccount/Basic_Information/Access_Overview.html)，微信公众号校验字符串的生成算法是：
1）将token、timestamp、nonce三个参数进行字典序排序  
2）将三个参数字符串拼接成一个字符串进行sha1加密
校验规则是： 公众号传入的校验字符串与本地生成的校验字符串必须保持一致。这样上面的代码逻辑就比较清晰了。

<div class="alert alert-success" role="alert">
<p> 微信公众号Token可以从 公众号后台/设置开发/基本配置 下获得</p>
<p> 服务器地址：云函数编辑框右上角有个发布按钮，点击发布后在发布按钮旁边的文本框中就是远程访问地址</p>
<p> 这里无需等待云函数编辑完毕才能发布，但是在生成好TOKEN后保存配置时，微信公众号后台会请求远程服务进行服务鉴权。如果这时鉴权逻辑没有写好，公众号的服务器配置将无法保存。 </p>
<p> 所以这里建议使用生成的token先把laf云函数的鉴权逻辑写好</p>
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-10.jpg" />
<img src="https://melon-note-1304191985.cos.ap-beijing.myqcloud.com/note/wechat-bot-11.jpg" />
</div>


<span class="label label-success">函数上下文对象</span>

```ts
export async function main(ctx: FunctionContext) {
  
}
```
通过阅读laf源码，可以看到FunctionContext对象定义如下：

```ts
/**
 * ctx passed to function
 */
export interface FunctionContext {
  files?: File[]
  headers?: IncomingHttpHeaders
  query?: any,
  body?: any,
  params?: any,
  auth?: any,
  requestId?: string,
  method?: string,
  response?: Response,
  __function_name?: string
}
```
这个上下文的构造方式为：

```ts
const ctx: FunctionContext = {
      query: req.query,
      files: req.files as any,
      body: req.body,
      headers: req.headers,
      method: isTrigger ? 'trigger' : req.method,
      auth: req['auth'],
      user: req.user,
      requestId,
      request: req,
      response: res,
      __function_name: func.name,
    }
```
可以看到，上下文中的大部分属性都是通过req构造出来的，那么req是什么？我们继续往下挖

```ts
// 执行云函数 laf/runtimes/nodejs/src/handler/invoke-func.ts 
export async function handleInvokeFunction(req: IRequest, res: Response) {
  // intercept the request, skip websocket request
  if (false === req.method.startsWith('WebSocket:')) {
    const passed = await invokeInterceptor(req, res)
    if (passed === false) return
  }

// laf/runtimes/nodejs/src/support/types.ts 
import { Request } from 'express'
export interface IRequest extends Request {
  user?: any
  requestId?: string
  [key: string]: any
}
```
如上所示，这个req本质上来源于express框架封装的请求对象。Express中的Request对象是一个表示HTTP请求的对象，它包含了请求的查询字符串，参数，内容，HTTP头部等属性
request对象有一些常用的属性和方法，例如：
req.app：访问express的实例。
req.baseUrl：获取路由当前安装的URL路径。
req.body：获取请求体。
req.cookies：获取请求中的cookie。
req.hostname：获取主机名。
req.method：获取请求方法（GET, POST等。
req.params：获取路由参数。
req.query：获取查询字符串参数。
req.url：获取请求的URL。
req.get(field)：获取指定的HTTP请求头。
req.param(name)：获取命名的路由参数或查询字符串参数。

其中对于req.query，指的是获取请求URL中的参数，例如：
``` ts
// GET /search?q=tobi+ferret
console.dir(req.query.q)
// => 'tobi ferret'
```


回到云函数主体
```ts
  const { signature, timestamp, nonce, echostr } = ctx.query;
  const { fromusername, tousername, content, msgtype } = ctx.body.xml;
```
通过上面我们知道，ctx.query来自于公众号请求url中的附带参数，ctx.body是公众号请求过来的实际数据。那么参考[公众号开发文档](https://developers.weixin.qq.com/doc/oplatform/Third-party_Platforms/2.0/api/Before_Develop/Message_encryption_and_decryption.html)。服务器收到公众号的消息体格式如下：

```xml
<xml>
  <ToUserName><![CDATA[toUser]]></ToUserName>
  <FromUserName><![CDATA[fromUser]]></FromUserName>
  <CreateTime>12345678</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[你好]]></Content>
</xml>
```
> 从上面分析也可以得到，微信公众号在与其他服务器交互时，鉴权信息会通过url带入，数据信息通过body带入。

<span class="label label-success">调用openAI获取响应</span>

```ts
// 创建ChatGPTAPI实例
const api = new ChatGPTAPI({ apiKey: cloud.env.CHAT_GPT_API_KEY })// 这个apikey要从openai官网获取
// 发送消息并获取回复
const response = await api.sendMessage(content[0])
const message = response.text.trim();
```
<span class="label label-success">响应公众号</span>

```ts
// 构造回复的XML对象
const xmlObj = {
    xml: {
    ToUserName: { '#text': fromusername[0] },
    FromUserName: { '#text': tousername[0] },
    CreateTime: { '#text': new Date().getTime() },
    MsgType: { '#text': 'text' },
    Content: { '#text': message }
    }
};
// 转换为XML字符串并返回
const xmlStr = create(xmlObj).end({ prettyPrint: true });
return xmlStr;
```

### 服务部署和测试
1. 发布laf函数
2. 启用公众号服务器配置
3. 到微信公众号发送消息，查看公众号是否可以正常响应

