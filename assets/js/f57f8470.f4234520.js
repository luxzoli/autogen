"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[402],{58075:(e,t,s)=>{s.r(t),s.d(t,{assets:()=>a,contentTitle:()=>o,default:()=>p,frontMatter:()=>i,metadata:()=>c,toc:()=>l});var n=s(74848),r=s(28453);const i={sidebar_label:"text_compressors",title:"agentchat.contrib.capabilities.text_compressors"},o=void 0,c={id:"reference/agentchat/contrib/capabilities/text_compressors",title:"agentchat.contrib.capabilities.text_compressors",description:"TextCompressor",source:"@site/docs/reference/agentchat/contrib/capabilities/text_compressors.md",sourceDirName:"reference/agentchat/contrib/capabilities",slug:"/reference/agentchat/contrib/capabilities/text_compressors",permalink:"/autogen/docs/reference/agentchat/contrib/capabilities/text_compressors",draft:!1,unlisted:!1,editUrl:"https://github.com/microsoft/autogen/edit/main/website/docs/reference/agentchat/contrib/capabilities/text_compressors.md",tags:[],version:"current",frontMatter:{sidebar_label:"text_compressors",title:"agentchat.contrib.capabilities.text_compressors"},sidebar:"referenceSideBar",previous:{title:"teachability",permalink:"/autogen/docs/reference/agentchat/contrib/capabilities/teachability"},next:{title:"transform_messages",permalink:"/autogen/docs/reference/agentchat/contrib/capabilities/transform_messages"}},a={},l=[{value:"TextCompressor",id:"textcompressor",level:2},{value:"compress_text",id:"compress_text",level:3},{value:"LLMLingua",id:"llmlingua",level:2},{value:"__init__",id:"__init__",level:3}];function d(e){const t={code:"code",em:"em",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(t.h2,{id:"textcompressor",children:"TextCompressor"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-python",children:"class TextCompressor(Protocol)\n"})}),"\n",(0,n.jsx)(t.p,{children:"Defines a protocol for text compression to optimize agent interactions."}),"\n",(0,n.jsx)(t.h3,{id:"compress_text",children:"compress_text"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-python",children:"def compress_text(text: str, **compression_params) -> Dict[str, Any]\n"})}),"\n",(0,n.jsx)(t.p,{children:"This method takes a string as input and returns a dictionary containing the compressed text and other\nrelevant information. The compressed text should be stored under the 'compressed_text' key in the dictionary.\nTo calculate the number of saved tokens, the dictionary should include 'origin_tokens' and 'compressed_tokens' keys."}),"\n",(0,n.jsx)(t.h2,{id:"llmlingua",children:"LLMLingua"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-python",children:"class LLMLingua()\n"})}),"\n",(0,n.jsx)(t.p,{children:"Compresses text messages using LLMLingua for improved efficiency in processing and response generation."}),"\n",(0,n.jsx)(t.p,{children:"NOTE: The effectiveness of compression and the resultant token savings can vary based on the content of the messages\nand the specific configurations used for the PromptCompressor."}),"\n",(0,n.jsx)(t.h3,{id:"__init__",children:"__init__"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-python",children:'def __init__(prompt_compressor_kwargs: Dict = dict(\n    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",\n    use_llmlingua2=True,\n    device_map="cpu",\n),\n             structured_compression: bool = False) -> None\n'})}),"\n",(0,n.jsxs)(t.p,{children:[(0,n.jsx)(t.strong,{children:"Arguments"}),":"]}),"\n",(0,n.jsxs)(t.ul,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"prompt_compressor_kwargs"})," ",(0,n.jsx)(t.em,{children:"dict"}),' - A dictionary of keyword arguments for the PromptCompressor. Defaults to a\ndictionary with model_name set to "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",\nuse_llmlingua2 set to True, and device_map set to "cpu".']}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"structured_compression"})," ",(0,n.jsx)(t.em,{children:"bool"})," - A flag indicating whether to use structured compression. If True, the\nstructured_compress_prompt method of the PromptCompressor is used. Otherwise, the compress_prompt method\nis used. Defaults to False.\ndictionary."]}),"\n"]}),"\n",(0,n.jsxs)(t.p,{children:[(0,n.jsx)(t.strong,{children:"Raises"}),":"]}),"\n",(0,n.jsxs)(t.ul,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"ImportError"})," - If the llmlingua library is not installed."]}),"\n"]})]})}function p(e={}){const{wrapper:t}={...(0,r.R)(),...e.components};return t?(0,n.jsx)(t,{...e,children:(0,n.jsx)(d,{...e})}):d(e)}},28453:(e,t,s)=>{s.d(t,{R:()=>o,x:()=>c});var n=s(96540);const r={},i=n.createContext(r);function o(e){const t=n.useContext(i);return n.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function c(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:o(e.components),n.createElement(i.Provider,{value:t},e.children)}}}]);