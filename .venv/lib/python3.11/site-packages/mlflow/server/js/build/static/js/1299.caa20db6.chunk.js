"use strict";(self.webpackChunk_mlflow_mlflow=self.webpackChunk_mlflow_mlflow||[]).push([[1299],{91299:function(e,n,a){a.d(n,{r:function(){return J}});var t=a(68248),o=a(76118),i=a(27299),r=a(27757),s=a(46709),l=a(91105);const d="compareToRunUuid",u=()=>{var e;const[n,a]=(0,l.ok)();return[null!==(e=n.get(d))&&void 0!==e?e:void 0,(0,s.useCallback)((e=>{a((n=>void 0===e?(n.delete(d),n):(n.set(d,e),n)))}),[a])]};var c=a(11522),m=a(66916),p=a(49620),g=a(42747),v=a(60284),h=a(84021),f=a(67063),y=a(73408);var T={name:"a41n9l",styles:"justify-content:flex-start !important"},x={name:"0",styles:""},I={name:"bcffy2",styles:"display:flex;align-items:center;justify-content:space-between"},b={name:"fhxb3m",styles:"display:flex;flex-direction:row;align-items:center"},C={name:"a41n9l",styles:"justify-content:flex-start !important"};const Y=({experimentId:e,currentRunUuid:n,setCompareToRunUuid:a,compareToRunUuid:o,setCurrentRunUuid:l})=>{const{theme:d}=(0,r.u)(),u=(0,g.tz)(),c=(0,p.LE)(),{runInfos:Y}=(0,h.Xz)(e),U=(0,s.useMemo)((()=>{if(Y)return Y.map((e=>{var n;return{key:e.runUuid,value:null!==(n=e.runName)&&void 0!==n?n:e.runUuid}})).filter((e=>e.key))}),[Y]),w=(0,s.useMemo)((()=>{if(Y)return Y.filter((e=>e.runUuid!==n)).map((e=>{var n;return{key:e.runUuid,value:null!==(n=e.runName)&&void 0!==n?n:e.runUuid}})).filter((e=>Boolean(e.key)))}),[Y,n]),S=null===Y||void 0===Y?void 0:Y.find((e=>e.runUuid===n)),R=null===Y||void 0===Y?void 0:Y.find((e=>e.runUuid===o)),N=(0,s.useCallback)((n=>{const a=v.Ay.getRunPageRoute(e,n)+"/evaluations",t=new URLSearchParams(window.location.search),o=new URL(a,window.location.origin);t.forEach(((e,n)=>{o.searchParams.set(n,e)})),window.location.href=o.toString()}),[e]),D=null!==l&&void 0!==l?l:N;return n?(0,y.FD)("div",{css:(0,t.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center"},""),children:[(0,y.Y)("div",{css:(0,t.AH)({display:"flex",alignItems:"center",justifyContent:"flex-start",gap:d.spacing.sm},""),children:(0,y.FD)(i.DialogCombobox,{componentId:h.WB,id:"compare-to-run-combobox",value:n?[n]:void 0,children:[(0,y.Y)(i.DialogComboboxCustomButtonTriggerWrapper,{children:(0,y.Y)(r.B,{endIcon:(0,y.Y)(i.ChevronDownIcon,{}),componentId:"mlflow.evaluations_review.table_ui.compare_to_run_button",css:T,children:(0,y.FD)("div",{css:(0,t.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center",fontSize:`${d.typography.fontSizeSm}px !important`},""),children:[(0,y.Y)(f.E,{color:c(n)}),null!==S&&void 0!==S&&S.runName?(0,y.Y)(r.T.Hint,{children:null===S||void 0===S?void 0:S.runName}):u.formatMessage({id:"PUQxu5",defaultMessage:"Select baseline run"})]})})}),(0,y.Y)(i.DialogComboboxContent,{children:(0,y.Y)(i.DialogComboboxOptionList,{children:(U||[]).map(((e,a)=>(0,y.Y)(i.DialogComboboxOptionListSelectItem,{value:e.value,onChange:n=>D(e.key),checked:e.key===n,children:(0,y.FD)("div",{css:(0,t.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center"},""),children:[(0,y.Y)(f.E,{color:c(e.key)}),e.value]})},a)))})})]})}),(0,y.Y)("span",{css:x,children:u.formatMessage({id:"iYmFCZ",defaultMessage:"compare to"})}),a&&(0,y.Y)("div",{css:I,children:(0,y.FD)("div",{css:b,children:[(0,y.FD)(i.DialogCombobox,{componentId:h.WB,id:"compare-to-run-combobox",value:o?[o]:void 0,children:[(0,y.Y)(i.DialogComboboxCustomButtonTriggerWrapper,{children:(0,y.Y)(r.B,{endIcon:(0,y.Y)(i.ChevronDownIcon,{}),componentId:"mlflow.evaluations_review.table_ui.compare_to_run_button",css:C,children:(0,y.Y)("div",{css:(0,t.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center",fontSize:`${d.typography.fontSizeSm}px !important`},""),children:null!==R&&void 0!==R&&R.runName?(0,y.FD)(y.FK,{children:[(0,y.Y)(f.E,{color:c(o)}),(0,y.Y)(r.T.Hint,{children:null===R||void 0===R?void 0:R.runName})]}):(0,y.Y)("span",{css:(0,t.AH)({color:d.colors.textPlaceholder},""),children:u.formatMessage({id:"XkpMf+",defaultMessage:"baseline run"})})})})}),(0,y.Y)(i.DialogComboboxContent,{children:(0,y.Y)(i.DialogComboboxOptionList,{children:(w||[]).map(((e,n)=>(0,y.Y)(i.DialogComboboxOptionListSelectItem,{value:e.value,onChange:n=>a(e.key),checked:e.key===o,children:(0,y.FD)("div",{css:(0,t.AH)({display:"flex",gap:d.spacing.sm,alignItems:"center"},""),children:[(0,y.Y)(f.E,{color:c(e.key)}),e.value]})},n)))})})]}),(null===R||void 0===R?void 0:R.runName)&&(0,y.Y)(m.X,{"aria-hidden":"false",css:(0,t.AH)({color:d.colors.textPlaceholder,fontSize:d.typography.fontSizeSm,marginLeft:d.spacing.sm,":hover":{color:d.colors.actionTertiaryTextHover}},""),role:"button",onClick:()=>{a(void 0)},onPointerDownCapture:e=>{e.stopPropagation()}})]})})]}):(0,y.Y)(y.FK,{})};var U=a(31655),w=a(69986),S=a(26765);const R=e=>(0,s.useMemo)((()=>e?(0,o.intersection)((0,S.T)(e),[h.o8.Evaluations,h.o8.Metrics,h.o8.Assessments]):[]),[e]);var N=a(33656),D=a(88525),E=a(40724),k=a(56530),A=a(5690),M=a(38232),F=a(65765),_=a(43233);const L="_assessments.json",H=()=>{const e=(0,k.wA)(),[n,a]=(0,s.useState)(!1);return{savePendingAssessments:(0,s.useCallback)((async(n,t,i)=>{try{a(!0);const r=await(async e=>{const n=(0,F.To)(L,e),a=await(0,F.Up)(n).then((e=>JSON.parse(e)));if(!(0,o.isArray)(a.data)||!(0,o.isArray)(a.columns))throw new Error("Artifact is malformed and/or not valid JSON");return a})(n),s=((e,n)=>n.map((n=>{var a,t,o;return[e,n.name,{source_type:null===(a=n.source)||void 0===a?void 0:a.sourceType,source_id:null===(t=n.source)||void 0===t?void 0:t.sourceId,source_metadata:null===(o=n.source)||void 0===o?void 0:o.metadata},n.timestamp||null,n.booleanValue||null,n.numericValue||null,n.stringValue||null,n.rationale||null,n.metadata||null,null,null]})))(t,i),l=((e,n,a)=>{const t=(0,M.G4)(L,n),i=a.map((({name:e,source:n})=>({name:e,source:n?{source_type:n.sourceType,source_id:n.sourceId,source_metadata:n.metadata}:void 0}))),r=t.entries.filter((({evaluation_id:n,name:a,source:t})=>e===n&&i.find((e=>(0,o.isEqual)({name:a,source:t},e))))).map((e=>t.entries.indexOf(e)));return n.data.filter(((e,n)=>!r.includes(n)))})(t,r,i);await e((0,A.Of)(n,L,{columns:r.columns,data:[...l,...s]})),e({type:(0,_.ec)(A.So),payload:(0,M.G4)(L,{columns:r.columns,data:[...l,...s]}),meta:{runUuid:n,artifactPath:L}})}catch(r){throw c.A.logErrorAndNotifyUser(r.message||r),r}finally{a(!1)}}),[e]),isSaving:n}};var $=a(81641);const O=$.J1`
  query SearchRuns($data: MlflowSearchRunsInput!) {
    mlflowSearchRuns(input: $data) {
      apiError {
        helpUrl
        code
        message
      }
      runs {
        info {
          runName
          status
          runUuid
          experimentId
          artifactUri
          endTime
          lifecycleStage
          startTime
          userId
        }
        experiment {
          experimentId
          name
          tags {
            key
            value
          }
          artifactLocation
          lifecycleStage
          lastUpdateTime
        }
        data {
          metrics {
            key
            value
            step
            timestamp
          }
          params {
            key
            value
          }
          tags {
            key
            value
          }
        }
        inputs {
          datasetInputs {
            dataset {
              digest
              name
              profile
              schema
              source
              sourceType
            }
            tags {
              key
              value
            }
          }
          modelInputs {
            modelId
          }
        }
        outputs {
          modelOutputs {
            modelId
            step
          }
        }
        modelVersions {
          version
          name
          creationTimestamp
          status
          source
        }
      }
    }
  }
`,P=({filter:e,experimentIds:n,disabled:a=!1})=>{var t,i,r;const{data:s,loading:l,error:d,refetch:u}=(0,$.IT)(O,{variables:{data:{filter:e,experimentIds:n}},skip:a});return{loading:l,data:(0,o.first)(null!==(t=null===s||void 0===s||null===(i=s.mlflowSearchRuns)||void 0===i?void 0:i.runs)&&void 0!==t?t:[]),refetchRun:u,apolloError:d,apiError:null===s||void 0===s||null===(r=s.mlflowSearchRuns)||void 0===r?void 0:r.apiError}};var j={name:"r3950p",styles:"flex:1;display:flex;justify-content:center;align-items:center"};const B=({experimentId:e,runUuid:n,runTags:a,runDisplayName:o,data:s})=>{const{theme:l}=(0,r.u)(),d=R(a),c=0===(null===s||void 0===s?void 0:s.length),[m,p]=u(),g=(0,N.N9)(),v=H(),{data:f,displayName:T,loading:x}=z(e,m,d);if(x)return(0,y.Y)(i.LegacySkeleton,{});const I=e=>e.filter((e=>e.type===h.$6.ASSESSMENT||e.type===h.$6.INPUT||e.type===h.$6.TRACE_INFO&&[h.tj,h.$W,h.Pn].includes(e.id)));return c?(0,y.Y)("div",{css:j,children:(0,y.Y)(i.Empty,{title:(0,y.Y)(E.A,{id:"NqqMPs",defaultMessage:"No evaluation tables logged"}),description:null})}):(0,y.FD)("div",{css:(0,t.AH)({marginTop:l.spacing.sm,width:"100%",overflowY:"hidden"},""),children:[(0,y.Y)("div",{css:(0,t.AH)({width:"100%",padding:`${l.spacing.xs}px 0`},""),children:(0,y.Y)(Y,{experimentId:e,currentRunUuid:n,compareToRunUuid:m,setCompareToRunUuid:p})}),(()=>{const a={experimentId:e,currentRunDisplayName:o,currentEvaluationResults:s||[],compareToEvaluationResults:f,runUuid:n,compareToRunUuid:m,compareToRunDisplayName:T,compareToRunLoading:x,saveAssessmentsQuery:v,getTrace:w.Rb,initialSelectedColumns:I};return(0,y.Y)(h.tU,{makeHtml:g,children:(0,y.Y)(h.js,{...a})})})()]})},z=(e,n,a)=>{const{data:t,isLoading:i}=(0,h.Ie)({runUuid:n||"",artifacts:a},{disabled:(0,o.isNil)(n)}),{data:r,loading:s}=P({experimentIds:[e],filter:`attributes.runId = "${n}"`,disabled:(0,o.isNil)(n)});return{data:t,displayName:c.A.getRunDisplayName(null===r||void 0===r?void 0:r.info,n),loading:i||s}};var Q=a(39595),W=a(82636),V=a(27462),q=a(52855);const K=({children:e,makeHtmlFromMarkdown:n,experimentId:a})=>(0,y.Y)(h.tU,{makeHtml:n,children:e});var X={name:"1nxh63r",styles:"overflow-y:hidden;height:100%;display:flex;flex-direction:column"};const G=({experimentId:e,runUuid:n,runDisplayName:a,setCurrentRunUuid:o})=>{const{theme:i}=(0,r.u)(),l=(0,N.N9)(),[d,c]=u(),m=(0,s.useMemo)((()=>[(0,h.$U)(e)]),[e]),g=w.Uv,v=!1,{assessmentInfos:f,allColumns:T,totalCount:x,evaluatedTraces:I,otherEvaluatedTraces:b,isLoading:C,error:S,tableFilterOptions:R}=(0,h.KW)({locations:m,runUuid:n,otherRunUuid:d,disabled:v}),[E,k]=(0,s.useState)(""),[A,M]=(0,h.R7)(),F=(0,p.LE)(),_=(0,Q.jE)(),L=(0,s.useCallback)((e=>{const{responseHasContent:n,inputHasContent:a,tokensHasContent:t}=(0,W.l)(I.concat(b));return e.filter((e=>e.type===h.$6.ASSESSMENT||e.type===h.$6.EXPECTATION||a&&e.type===h.$6.INPUT||n&&e.type===h.$6.TRACE_INFO&&e.id===h.Rl||t&&e.type===h.$6.TRACE_INFO&&e.id===h.YO||e.type===h.$6.TRACE_INFO&&[h.XQ,h.tj,h.$W].includes(e.id)))}),[I,b]),{selectedColumns:H,toggleColumns:$,setSelectedColumns:O}=(0,h.K0)(e,T,L,n),[P,j]=(0,h.GY)(H),{data:B,isLoading:z,isFetching:G,error:J,refetchMlflowTraces:ne}=(0,h.Zn)({locations:m,currentRunDisplayName:a,searchQuery:E,filters:A,runUuid:n,tableSort:P,disabled:v}),{data:ae,displayName:te,loading:oe}=ee({experimentId:e,traceLocations:m,compareToRunUuid:d,isQueryDisabled:v}),ie=(0,s.useMemo)((()=>({currentCount:null===B||void 0===B?void 0:B.length,logCountLoading:z,totalCount:x,maxAllowedCount:(0,U.pR)()})),[B,z,x]),{showEditTagsModalForTrace:re,EditTagsModal:se}=(0,D.$)({onSuccess:()=>(0,h.BL)({queryClient:_}),existingTagKeys:(0,h.d9)(B||[])}),le=(0,V.F)({traceSearchLocations:m}),{showExportTracesToDatasetsModal:de,setShowExportTracesToDatasetsModal:ue,renderExportTracesToDatasetsModal:ce}=(0,q.c)({experimentId:e}),me=(0,s.useMemo)((()=>({deleteTracesAction:le,exportToEvals:{showExportTracesToDatasetsModal:de,setShowExportTracesToDatasetsModal:ue,renderExportTracesToDatasetsModal:ce},editTags:{showEditTagsModalForTrace:re,EditTagsModal:se}})),[le,de,ue,ce,re,se]),pe=z||oe;return C?(0,y.Y)(Z,{}):S?(0,y.Y)("div",{children:(0,y.Y)("pre",{children:String(S)})}):(0,y.FD)("div",{css:(0,t.AH)({marginTop:i.spacing.sm,width:"100%",overflowY:"hidden"},""),children:[(0,y.Y)("div",{css:(0,t.AH)({width:"100%",padding:`${i.spacing.xs}px 0`},""),children:(0,y.Y)(Y,{experimentId:e,currentRunUuid:n,compareToRunUuid:d,setCompareToRunUuid:c,setCurrentRunUuid:o})}),(0,y.Y)(h.sG,{children:(0,y.FD)("div",{css:X,children:[(0,y.Y)(h.w_,{experimentId:e,searchQuery:E,setSearchQuery:k,filters:A,setFilters:M,assessmentInfos:f,countInfo:ie,traceActions:me,tableSort:P,setTableSort:j,allColumns:T,selectedColumns:H,setSelectedColumns:O,toggleColumns:$,traceInfos:B,tableFilterOptions:R}),pe?(0,y.Y)(Z,{}):J?(0,y.Y)("div",{children:(0,y.Y)("pre",{children:String(J)})}):(0,y.Y)(K,{makeHtmlFromMarkdown:l,experimentId:e,children:(0,y.Y)(h._p,{experimentId:e,currentRunDisplayName:a,compareToRunDisplayName:te,compareToRunUuid:d,getTrace:g,getRunColor:F,assessmentInfos:f,setFilters:M,filters:A,selectedColumns:H,allColumns:T,tableSort:P,currentTraceInfoV3:B||[],compareToTraceInfoV3:ae,onTraceTagsEdit:re,displayLoadingOverlay:!1})}),se]})})]})},J=({experimentId:e,experiment:n,runUuid:a,runTags:t,runDisplayName:i,setCurrentRunUuid:r})=>{const s=R(t),l=Boolean(a),{data:d,isLoading:u}=(0,h.Ie)({runUuid:a||"",artifacts:s||void 0},{disabled:!l});return u?(0,y.Y)(Z,{}):!(0,o.isNil)(d)&&d.length>0?(0,y.Y)(B,{experimentId:e,runUuid:a,runDisplayName:i,data:d,runTags:t}):(0,y.Y)(G,{experimentId:e,runUuid:a,runDisplayName:i,setCurrentRunUuid:r})},Z=()=>{const{theme:e}=(0,r.u)();return(0,y.Y)("div",{css:(0,t.AH)({display:"block",marginTop:e.spacing.md,height:"100%",width:"100%"},""),children:[...Array(10).keys()].map((e=>(0,y.Y)(i.ParagraphSkeleton,{label:"Loading...",seed:`s-${e}`},e)))})},ee=e=>{const{compareToRunUuid:n,experimentId:a,traceLocations:t,isQueryDisabled:i}=e,{data:r,isLoading:s}=(0,h.Zn)({locations:t,currentRunDisplayName:void 0,runUuid:n,disabled:(0,o.isNil)(n)||i}),{data:l,loading:d}=P({experimentIds:[a],filter:`attributes.runId = "${n}"`,disabled:(0,o.isNil)(n)});return{data:r,displayName:c.A.getRunDisplayName(null===l||void 0===l?void 0:l.info,n),loading:s||d}}}}]);