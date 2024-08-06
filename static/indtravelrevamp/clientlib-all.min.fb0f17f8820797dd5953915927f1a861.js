/*! jQuery UI - v1.12.1 - 2016-09-14
* http://jqueryui.com
* Includes: widget.js, position.js, data.js, disable-selection.js, effect.js, effects/effect-blind.js, effects/effect-bounce.js, effects/effect-clip.js, effects/effect-drop.js, effects/effect-explode.js, effects/effect-fade.js, effects/effect-fold.js, effects/effect-highlight.js, effects/effect-puff.js, effects/effect-pulsate.js, effects/effect-scale.js, effects/effect-shake.js, effects/effect-size.js, effects/effect-slide.js, effects/effect-transfer.js, focusable.js, form-reset-mixin.js, jquery-1-7.js, keycode.js, labels.js, scroll-parent.js, tabbable.js, unique-id.js, widgets/accordion.js, widgets/autocomplete.js, widgets/button.js, widgets/checkboxradio.js, widgets/controlgroup.js, widgets/datepicker.js, widgets/dialog.js, widgets/draggable.js, widgets/droppable.js, widgets/menu.js, widgets/mouse.js, widgets/progressbar.js, widgets/resizable.js, widgets/selectable.js, widgets/selectmenu.js, widgets/slider.js, widgets/sortable.js, widgets/spinner.js, widgets/tabs.js, widgets/tooltip.js
* Copyright jQuery Foundation and other contributors; Licensed MIT */
(function(a){"function"==typeof define&&define.amd?define(["jquery"],a):a(jQuery)
})(function(k){function G(a){for(var c=a.css("visibility");
"inherit"===c;
){a=a.parent(),c=a.css("visibility")
}return"hidden"!==c
}function C(c){for(var d,a;
c.length&&c[0]!==document;
){if(d=c.css("position"),("absolute"===d||"relative"===d||"fixed"===d)&&(a=parseInt(c.css("zIndex"),10),!isNaN(a)&&0!==a)){return a
}c=c.parent()
}return 0
}function q(){this._curInst=null,this._keyEvent=!1,this._disabledInputs=[],this._datepickerShowing=!1,this._inDialog=!1,this._mainDivId="ui-datepicker-div",this._inlineClass="ui-datepicker-inline",this._appendClass="ui-datepicker-append",this._triggerClass="ui-datepicker-trigger",this._dialogClass="ui-datepicker-dialog",this._disableClass="ui-datepicker-disabled",this._unselectableClass="ui-datepicker-unselectable",this._currentClass="ui-datepicker-current-day",this._dayOverClass="ui-datepicker-days-cell-over",this.regional=[],this.regional[""]={closeText:"Done",prevText:"Prev",nextText:"Next",currentText:"Today",monthNames:["January","February","March","April","May","June","July","August","September","October","November","December"],monthNamesShort:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],dayNames:["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],dayNamesShort:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],dayNamesMin:["Su","Mo","Tu","We","Th","Fr","Sa"],weekHeader:"Wk",dateFormat:"mm/dd/yy",firstDay:0,isRTL:!1,showMonthAfterYear:!1,yearSuffix:""},this._defaults={showOn:"focus",showAnim:"fadeIn",showOptions:{},defaultDate:null,appendText:"",buttonText:"...",buttonImage:"",buttonImageOnly:!1,hideIfNoPrevNext:!1,navigationAsDateFormat:!1,gotoCurrent:!1,changeMonth:!1,changeYear:!1,yearRange:"c-10:c+10",showOtherMonths:!1,selectOtherMonths:!1,showWeek:!1,calculateWeek:this.iso8601Week,shortYearCutoff:"+10",minDate:null,maxDate:null,duration:"fast",beforeShowDay:null,beforeShow:null,onSelect:null,onChangeMonthYear:null,onClose:null,numberOfMonths:1,showCurrentAtPos:0,stepMonths:1,stepBigMonths:12,altField:"",altFormat:"",constrainInput:!0,showButtonPanel:!1,autoSize:!1,disabled:!1},k.extend(this._defaults,this.regional[""]),this.regional.en=k.extend(!0,{},this.regional[""]),this.regional["en-US"]=k.extend(!0,{},this.regional.en),this.dpDiv=z(k("<div id='"+this._mainDivId+"' class='ui-datepicker ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>"))
}function z(c){var a="button, .ui-datepicker-prev, .ui-datepicker-next, .ui-datepicker-calendar td a";
return c.on("mouseout",a,function(){k(this).removeClass("ui-state-hover"),-1!==this.className.indexOf("ui-datepicker-prev")&&k(this).removeClass("ui-datepicker-prev-hover"),-1!==this.className.indexOf("ui-datepicker-next")&&k(this).removeClass("ui-datepicker-next-hover")
}).on("mouseover",a,y)
}function y(){k.datepicker._isDisabledDatepicker(A.inline?A.dpDiv.parent()[0]:A.input[0])||(k(this).parents(".ui-datepicker-calendar").find("a").removeClass("ui-state-hover"),k(this).addClass("ui-state-hover"),-1!==this.className.indexOf("ui-datepicker-prev")&&k(this).addClass("ui-datepicker-prev-hover"),-1!==this.className.indexOf("ui-datepicker-next")&&k(this).addClass("ui-datepicker-next-hover"))
}function J(d,a){k.extend(d,a);
for(var c in a){null==a[c]&&(d[c]=a[c])
}return d
}function w(a){return function(){var c=this.element.val();
a.apply(this,arguments),this._refresh(),c!==this.element.val()&&this._trigger("change")
}
}k.ui=k.ui||{},k.ui.version="1.12.1";
var D=0,B=Array.prototype.slice;
k.cleanData=function(a){return function(d){var e,g,f;
for(f=0;
null!=(g=d[f]);
f++){try{e=k._data(g,"events"),e&&e.remove&&k(g).triggerHandler("remove")
}catch(c){}}a(d)
}
}(k.cleanData),k.widget=function(t,m,v){var f,d,u,c={},p=t.split(".")[0];
t=t.split(".")[1];
var g=p+"-"+t;
return v||(v=m,m=k.Widget),k.isArray(v)&&(v=k.extend.apply(null,[{}].concat(v))),k.expr[":"][g.toLowerCase()]=function(a){return !!k.data(a,g)
},k[p]=k[p]||{},f=k[p][t],d=k[p][t]=function(a,h){return this._createWidget?(arguments.length&&this._createWidget(a,h),void 0):new d(a,h)
},k.extend(d,f,{version:v.version,_proto:k.extend({},v),_childConstructors:[]}),u=new m,u.options=k.widget.extend({},u.options),k.each(v,function(h,a){return k.isFunction(a)?(c[h]=function(){function e(){return m.prototype[h].apply(this,arguments)
}function i(l){return m.prototype[h].apply(this,l)
}return function(){var n,l=this._super,r=this._superApply;
return this._super=e,this._superApply=i,n=a.apply(this,arguments),this._super=l,this._superApply=r,n
}
}(),void 0):(c[h]=a,void 0)
}),d.prototype=k.widget.extend(u,{widgetEventPrefix:f?u.widgetEventPrefix||t:t},c,{constructor:d,namespace:p,widgetName:t,widgetFullName:g}),f?(k.each(f._childConstructors,function(l,a){var h=a.prototype;
k.widget(h.namespace+"."+h.widgetName,d,a._proto)
}),delete f._childConstructors):m._childConstructors.push(d),k.widget.bridge(t,d),d
},k.widget.extend=function(g){for(var d,f,l=B.call(arguments,1),h=0,c=l.length;
c>h;
h++){for(d in l[h]){f=l[h][d],l[h].hasOwnProperty(d)&&void 0!==f&&(g[d]=k.isPlainObject(f)?k.isPlainObject(g[d])?k.widget.extend({},g[d],f):k.widget.extend({},f):f)
}}return g
},k.widget.bridge=function(d,a){var c=a.prototype.widgetFullName||d;
k.fn[d]=function(h){var g="string"==typeof h,e=B.call(arguments,1),f=this;
return g?this.length||"instance"!==h?this.each(function(){var l,m=k.data(this,c);
return"instance"===h?(f=m,!1):m?k.isFunction(m[h])&&"_"!==h.charAt(0)?(l=m[h].apply(m,e),l!==m&&void 0!==l?(f=l&&l.jquery?f.pushStack(l.get()):l,!1):void 0):k.error("no such method '"+h+"' for "+d+" widget instance"):k.error("cannot call methods on "+d+" prior to initialization; attempted to call method '"+h+"'")
}):f=void 0:(e.length&&(h=k.widget.extend.apply(null,[h].concat(e))),this.each(function(){var i=k.data(this,c);
i?(i.option(h||{}),i._init&&i._init()):k.data(this,c,new a(h,this))
})),f
}
},k.Widget=function(){},k.Widget._childConstructors=[],k.Widget.prototype={widgetName:"widget",widgetEventPrefix:"",defaultElement:"<div>",options:{classes:{},disabled:!1,create:null},_createWidget:function(c,a){a=k(a||this.defaultElement||this)[0],this.element=k(a),this.uuid=D++,this.eventNamespace="."+this.widgetName+this.uuid,this.bindings=k(),this.hoverable=k(),this.focusable=k(),this.classesElementLookup={},a!==this&&(k.data(a,this.widgetFullName,this),this._on(!0,this.element,{remove:function(d){d.target===a&&this.destroy()
}}),this.document=k(a.style?a.ownerDocument:a.document||a),this.window=k(this.document[0].defaultView||this.document[0].parentWindow)),this.options=k.widget.extend({},this.options,this._getCreateOptions(),c),this._create(),this.options.disabled&&this._setOptionDisabled(this.options.disabled),this._trigger("create",null,this._getCreateEventData()),this._init()
},_getCreateOptions:function(){return{}
},_getCreateEventData:k.noop,_create:k.noop,_init:k.noop,destroy:function(){var a=this;
this._destroy(),k.each(this.classesElementLookup,function(d,c){a._removeClass(c,d)
}),this.element.off(this.eventNamespace).removeData(this.widgetFullName),this.widget().off(this.eventNamespace).removeAttr("aria-disabled"),this.bindings.off(this.eventNamespace)
},_destroy:k.noop,widget:function(){return this.element
},option:function(g,d){var f,l,h,c=g;
if(0===arguments.length){return k.widget.extend({},this.options)
}if("string"==typeof g){if(c={},f=g.split("."),g=f.shift(),f.length){for(l=c[g]=k.widget.extend({},this.options[g]),h=0;
f.length-1>h;
h++){l[f[h]]=l[f[h]]||{},l=l[f[h]]
}if(g=f.pop(),1===arguments.length){return void 0===l[g]?null:l[g]
}l[g]=d
}else{if(1===arguments.length){return void 0===this.options[g]?null:this.options[g]
}c[g]=d
}}return this._setOptions(c),this
},_setOptions:function(a){var c;
for(c in a){this._setOption(c,a[c])
}return this
},_setOption:function(a,c){return"classes"===a&&this._setOptionClasses(c),this.options[a]=c,"disabled"===a&&this._setOptionDisabled(c),this
},_setOptionClasses:function(d){var a,c,f;
for(a in d){f=this.classesElementLookup[a],d[a]!==this.options.classes[a]&&f&&f.length&&(c=k(f.get()),this._removeClass(f,a),c.addClass(this._classes({element:c,keys:a,classes:d,add:!0})))
}},_setOptionDisabled:function(a){this._toggleClass(this.widget(),this.widgetFullName+"-disabled",null,!!a),a&&(this._removeClass(this.hoverable,null,"ui-state-hover"),this._removeClass(this.focusable,null,"ui-state-focus"))
},enable:function(){return this._setOptions({disabled:!1})
},disable:function(){return this._setOptions({disabled:!0})
},_classes:function(d){function a(g,l){var e,h;
for(h=0;
g.length>h;
h++){e=f.classesElementLookup[g[h]]||k(),e=d.add?k(k.unique(e.get().concat(d.element.get()))):k(e.not(d.element).get()),f.classesElementLookup[g[h]]=e,c.push(g[h]),l&&d.classes[g[h]]&&c.push(d.classes[g[h]])
}}var c=[],f=this;
return d=k.extend({element:this.element,classes:this.options.classes||{}},d),this._on(d.element,{remove:"_untrackClassesElement"}),d.keys&&a(d.keys.match(/\S+/g)||[],!0),d.extra&&a(d.extra.match(/\S+/g)||[]),c.join(" ")
},_untrackClassesElement:function(c){var a=this;
k.each(a.classesElementLookup,function(d,e){-1!==k.inArray(c.target,e)&&(a.classesElementLookup[d]=k(e.not(c.target).get()))
})
},_removeClass:function(c,d,a){return this._toggleClass(c,d,a,!1)
},_addClass:function(c,d,a){return this._toggleClass(c,d,a,!0)
},_toggleClass:function(c,f,a,d){d="boolean"==typeof d?d:a;
var h="string"==typeof c||null===c,g={extra:h?f:a,keys:h?c:f,element:h?this.element:c,add:d};
return g.element.toggleClass(this._classes(g),d),this
},_on:function(d,a,c){var g,f=this;
"boolean"!=typeof d&&(c=a,a=d,d=!1),c?(a=g=k(a),this.bindings=this.bindings.add(a)):(c=a,a=this.element,g=this.widget()),k.each(c,function(n,i){function o(){return d||f.options.disabled!==!0&&!k(this).hasClass("ui-state-disabled")?("string"==typeof i?f[i]:i).apply(f,arguments):void 0
}"string"!=typeof i&&(o.guid=i.guid=i.guid||o.guid||k.guid++);
var m=n.match(/^([\w:-]*)\s*(.*)$/),e=m[1]+f.eventNamespace,p=m[2];
p?g.on(e,p,o):a.on(e,o)
})
},_off:function(c,a){a=(a||"").split(" ").join(this.eventNamespace+" ")+this.eventNamespace,c.off(a).off(a),this.bindings=k(this.bindings.not(c).get()),this.focusable=k(this.focusable.not(c).get()),this.hoverable=k(this.hoverable.not(c).get())
},_delay:function(c,f){function a(){return("string"==typeof c?d[c]:c).apply(d,arguments)
}var d=this;
return setTimeout(a,f||0)
},_hoverable:function(a){this.hoverable=this.hoverable.add(a),this._on(a,{mouseenter:function(c){this._addClass(k(c.currentTarget),null,"ui-state-hover")
},mouseleave:function(c){this._removeClass(k(c.currentTarget),null,"ui-state-hover")
}})
},_focusable:function(a){this.focusable=this.focusable.add(a),this._on(a,{focusin:function(c){this._addClass(k(c.currentTarget),null,"ui-state-focus")
},focusout:function(c){this._removeClass(k(c.currentTarget),null,"ui-state-focus")
}})
},_trigger:function(g,d,f){var l,h,c=this.options[g];
if(f=f||{},d=k.Event(d),d.type=(g===this.widgetEventPrefix?g:this.widgetEventPrefix+g).toLowerCase(),d.target=this.element[0],h=d.originalEvent){for(l in h){l in d||(d[l]=h[l])
}}return this.element.trigger(d,f),!(k.isFunction(c)&&c.apply(this.element[0],[d].concat(f))===!1||d.isDefaultPrevented())
}},k.each({show:"fadeIn",hide:"fadeOut"},function(c,a){k.Widget.prototype["_"+c]=function(e,h,g){"string"==typeof h&&(h={effect:h});
var d,f=h?h===!0||"number"==typeof h?a:h.effect||a:c;
h=h||{},"number"==typeof h&&(h={duration:h}),d=!k.isEmptyObject(h),h.complete=g,h.delay&&e.delay(h.delay),d&&k.effects&&k.effects.effect[f]?e[c](h):f!==c&&e[f]?e[f](h.duration,h.easing,g):e.queue(function(l){k(this)[c](),g&&g.call(e[0]),l()
})
}
}),k.widget,function(){function L(c,d,a){return[parseFloat(c[0])*(P.test(c[0])?d/100:1),parseFloat(c[1])*(P.test(c[1])?a/100:1)]
}function t(c,a){return parseInt(k.css(c,a),10)||0
}function Q(c){var a=c[0];
return 9===a.nodeType?{width:c.width(),height:c.height(),offset:{top:0,left:0}}:k.isWindow(a)?{width:c.width(),height:c.height(),offset:{top:c.scrollTop(),left:c.scrollLeft()}}:a.preventDefault?{width:0,height:0,offset:{top:a.pageY,left:a.pageX}}:{width:c.outerWidth(),height:c.outerHeight(),offset:c.offset()}
}var m,g=Math.max,O=Math.abs,f=/left|center|right/,v=/top|center|bottom/,p=/[\+\-]\d+(\.[\d]+)?%?/,N=/^\w+/,P=/%$/,M=k.fn.position;
k.position={scrollbarWidth:function(){if(void 0!==m){return m
}var d,a,c=k("<div style='display:block;position:absolute;width:50px;height:50px;overflow:hidden;'><div style='height:100px;width:auto;'></div></div>"),h=c.children()[0];
return k("body").append(c),d=h.offsetWidth,c.css("overflow","scroll"),a=h.offsetWidth,d===a&&(a=c[0].clientWidth),c.remove(),m=d-a
},getScrollInfo:function(d){var a=d.isWindow||d.isDocument?"":d.element.css("overflow-x"),c=d.isWindow||d.isDocument?"":d.element.css("overflow-y"),l="scroll"===a||"auto"===a&&d.width<d.element[0].scrollWidth,h="scroll"===c||"auto"===c&&d.height<d.element[0].scrollHeight;
return{width:h?k.position.scrollbarWidth():0,height:l?k.position.scrollbarWidth():0}
},getWithinInfo:function(d){var a=k(d||window),c=k.isWindow(a[0]),l=!!a[0]&&9===a[0].nodeType,h=!c&&!l;
return{element:a,isWindow:c,isDocument:l,offset:h?k(d).offset():{left:0,top:0},scrollLeft:a.scrollLeft(),scrollTop:a.scrollTop(),width:a.outerWidth(),height:a.outerHeight()}
}},k.fn.position=function(c){if(!c||!c.of){return M.apply(this,arguments)
}c=k.extend({},c);
var S,a,i,h,d,s,R=k(c.of),o=k.position.getWithinInfo(c.within),l=k.position.getScrollInfo(o),r=(c.collision||"flip").split(" "),e={};
return s=Q(R),R[0].preventDefault&&(c.at="left top"),a=s.width,i=s.height,h=s.offset,d=k.extend({},h),k.each(["my","at"],function(){var u,T,n=(c[this]||"").split(" ");
1===n.length&&(n=f.test(n[0])?n.concat(["center"]):v.test(n[0])?["center"].concat(n):["center","center"]),n[0]=f.test(n[0])?n[0]:"center",n[1]=v.test(n[1])?n[1]:"center",u=p.exec(n[0]),T=p.exec(n[1]),e[this]=[u?u[0]:0,T?T[0]:0],c[this]=[N.exec(n[0])[0],N.exec(n[1])[0]]
}),1===r.length&&(r[1]=r[0]),"right"===c.at[0]?d.left+=a:"center"===c.at[0]&&(d.left+=a/2),"bottom"===c.at[1]?d.top+=i:"center"===c.at[1]&&(d.top+=i/2),S=L(e.at,a,i),d.left+=S[0],d.top+=S[1],this.each(function(){var ab,u,V=k(this),U=V.outerWidth(),X=V.outerHeight(),W=t(this,"marginLeft"),aa=t(this,"marginTop"),Y=U+W+t(this,"marginRight")+l.width,T=X+aa+t(this,"marginBottom")+l.height,n=k.extend({},d),Z=L(e.my,V.outerWidth(),V.outerHeight());
"right"===c.my[0]?n.left-=U:"center"===c.my[0]&&(n.left-=U/2),"bottom"===c.my[1]?n.top-=X:"center"===c.my[1]&&(n.top-=X/2),n.left+=Z[0],n.top+=Z[1],ab={marginLeft:W,marginTop:aa},k.each(["left","top"],function(ad,ac){k.ui.position[r[ad]]&&k.ui.position[r[ad]][ac](n,{targetWidth:a,targetHeight:i,elemWidth:U,elemHeight:X,collisionPosition:ab,collisionWidth:Y,collisionHeight:T,offset:[S[0]+Z[0],S[1]+Z[1]],my:c.my,at:c.at,within:o,elem:V})
}),c.using&&(u=function(ae){var ah=h.left-n.left,ad=ah+a-U,af=h.top-n.top,ag=af+i-X,ac={target:{element:R,left:h.left,top:h.top,width:a,height:i},element:{element:V,left:n.left,top:n.top,width:U,height:X},horizontal:0>ad?"left":ah>0?"right":"center",vertical:0>ag?"top":af>0?"bottom":"middle"};
U>a&&a>O(ah+ad)&&(ac.horizontal="center"),X>i&&i>O(af+ag)&&(ac.vertical="middle"),ac.important=g(O(ah),O(ad))>g(O(af),O(ag))?"horizontal":"vertical",c.using.call(this,ae,ac)
}),V.offset(k.extend(n,{using:u}))
})
},k.ui.position={fit:{left:function(U,S){var u,V=S.within,d=V.isWindow?V.scrollLeft:V.offset.left,T=V.width,c=U.left-S.collisionPosition.marginLeft,R=d-c,o=c+S.collisionWidth-T-d;
S.collisionWidth>T?R>0&&0>=o?(u=U.left+R+S.collisionWidth-T-d,U.left+=R-u):U.left=o>0&&0>=R?d:R>o?d+T-S.collisionWidth:d:R>0?U.left+=R:o>0?U.left-=o:U.left=g(U.left-c,U.left)
},top:function(U,S){var u,V=S.within,d=V.isWindow?V.scrollTop:V.offset.top,T=S.within.height,c=U.top-S.collisionPosition.marginTop,R=d-c,o=c+S.collisionHeight-T-d;
S.collisionHeight>T?R>0&&0>=o?(u=U.top+R+S.collisionHeight-T-d,U.top+=R-u):U.top=o>0&&0>=R?d:R>o?d+T-S.collisionHeight:d:R>0?U.top+=R:o>0?U.top-=o:U.top=g(U.top-c,U.top)
}},flip:{left:function(ac,Y){var V,ad,T=Y.within,S=T.offset.left+T.scrollLeft,a=T.width,W=T.isWindow?T.scrollLeft:T.offset.left,U=ac.left-Y.collisionPosition.marginLeft,aa=U-W,ab=U+Y.collisionWidth-a-W,Z="left"===Y.my[0]?-Y.elemWidth:"right"===Y.my[0]?Y.elemWidth:0,R="left"===Y.at[0]?Y.targetWidth:"right"===Y.at[0]?-Y.targetWidth:0,X=-2*Y.offset[0];
0>aa?(V=ac.left+Z+R+X+Y.collisionWidth-a-S,(0>V||O(aa)>V)&&(ac.left+=Z+R+X)):ab>0&&(ad=ac.left-Y.collisionPosition.marginLeft+Z+R+X-W,(ad>0||ab>O(ad))&&(ac.left+=Z+R+X))
},top:function(ad,Z){var V,ae,T=Z.within,S=T.offset.top+T.scrollTop,a=T.height,W=T.isWindow?T.scrollTop:T.offset.top,U=ad.top-Z.collisionPosition.marginTop,ab=U-W,ac=U+Z.collisionHeight-a-W,aa="top"===Z.my[1],R=aa?-Z.elemHeight:"bottom"===Z.my[1]?Z.elemHeight:0,Y="top"===Z.at[1]?Z.targetHeight:"bottom"===Z.at[1]?-Z.targetHeight:0,X=-2*Z.offset[1];
0>ab?(ae=ad.top+R+Y+X+Z.collisionHeight-a-S,(0>ae||O(ab)>ae)&&(ad.top+=R+Y+X)):ac>0&&(V=ad.top-Z.collisionPosition.marginTop+R+Y+X-W,(V>0||ac>O(V))&&(ad.top+=R+Y+X))
}},flipfit:{left:function(){k.ui.position.flip.left.apply(this,arguments),k.ui.position.fit.left.apply(this,arguments)
},top:function(){k.ui.position.flip.top.apply(this,arguments),k.ui.position.fit.top.apply(this,arguments)
}}}
}(),k.ui.position,k.extend(k.expr[":"],{data:k.expr.createPseudo?k.expr.createPseudo(function(a){return function(c){return !!k.data(c,a)
}
}):function(d,a,c){return !!k.data(d,c[3])
}}),k.fn.extend({disableSelection:function(){var a="onselectstart" in document.createElement("div")?"selectstart":"mousedown";
return function(){return this.on(a+".ui-disableSelection",function(c){c.preventDefault()
})
}
}(),enableSelection:function(){return this.off(".ui-disableSelection")
}});
var I="ui-effects-",j="ui-effects-style",H="ui-effects-animated",x=k;
k.effects={effect:{}},function(V,Q){function N(c,f,a){var d=U[f.type]||{};
return null==c?a||!f.def?null:f.def:(c=d.floor?~~c:parseFloat(c),isNaN(c)?f.def:d.mod?(c+d.mod)%d.mod:0>c?0:c>d.max?d.max:c)
}function W(a){var c=M(),d=c._rgba=[];
return a=a.toLowerCase(),P(O,function(i,s){var f,p=s.re.exec(a),n=p&&s.parse(p),e=s.space||"rgba";
return n?(f=c[e](n),c[S[e].cache]=f[S[e].cache],d=c._rgba=f._rgba,!1):Q
}),d.length?("0,0,0,0"===d.join()&&V.extend(d,v.transparent),c):v[a]
}function L(c,d,a){return a=(a+1)%1,1>6*a?c+6*(d-c)*a:1>2*a?d:2>3*a?c+6*(d-c)*(2/3-a):c
}var v,T="backgroundColor borderBottomColor borderLeftColor borderRightColor borderTopColor color columnRuleColor outlineColor textDecorationColor textEmphasisColor",g=/^([\-+])=\s*(\d+\.?\d*)/,O=[{re:/rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,parse:function(a){return[a[1],a[2],a[3],a[4]]
}},{re:/rgba?\(\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,parse:function(a){return[2.55*a[1],2.55*a[2],2.55*a[3],a[4]]
}},{re:/#([a-f0-9]{2})([a-f0-9]{2})([a-f0-9]{2})/,parse:function(a){return[parseInt(a[1],16),parseInt(a[2],16),parseInt(a[3],16)]
}},{re:/#([a-f0-9])([a-f0-9])([a-f0-9])/,parse:function(a){return[parseInt(a[1]+a[1],16),parseInt(a[2]+a[2],16),parseInt(a[3]+a[3],16)]
}},{re:/hsla?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,space:"hsla",parse:function(a){return[a[1],a[2]/100,a[3]/100,a[4]]
}}],M=V.Color=function(d,a,c,f){return new V.Color.fn.parse(d,a,c,f)
},S={rgba:{props:{red:{idx:0,type:"byte"},green:{idx:1,type:"byte"},blue:{idx:2,type:"byte"}}},hsla:{props:{hue:{idx:0,type:"degrees"},saturation:{idx:1,type:"percent"},lightness:{idx:2,type:"percent"}}}},U={"byte":{floor:!0,max:255},percent:{max:1},degrees:{mod:360,floor:!0}},R=M.support={},m=V("<p>")[0],P=V.each;
m.style.cssText="background-color:rgba(1,1,1,.5)",R.rgba=m.style.backgroundColor.indexOf("rgba")>-1,P(S,function(a,c){c.cache="_"+a,c.props.alpha={idx:3,type:"percent",def:1}
}),M.fn=V.extend(M.prototype,{parse:function(s,c,i,f){if(s===Q){return this._rgba=[null,null,null,null],this
}(s.jquery||s.nodeType)&&(s=V(s).css(c),c=Q);
var e=this,o=V.type(s),l=this._rgba=[];
return c!==Q&&(s=[s,c,i,f],o="array"),"string"===o?this.parse(W(s)||v._default):"array"===o?(P(S.rgba.props,function(a,d){l[d.idx]=N(s[d.idx],d)
}),this):"object"===o?(s instanceof M?P(S,function(a,d){s[d.cache]&&(e[d.cache]=s[d.cache].slice())
}):P(S,function(d,a){var h=a.cache;
P(a.props,function(n,p){if(!e[h]&&a.to){if("alpha"===n||null==s[n]){return
}e[h]=a.to(e._rgba)
}e[h][p.idx]=N(s[n],p,!0)
}),e[h]&&0>V.inArray(null,e[h].slice(0,3))&&(e[h][3]=1,a.from&&(e._rgba=a.from(e[h])))
}),this):Q
},is:function(c){var a=M(c),d=!0,e=this;
return P(S,function(h,l){var f,i=a[l.cache];
return i&&(f=e[l.cache]||l.to&&l.to(e._rgba)||[],P(l.props,function(o,n){return null!=i[n.idx]?d=i[n.idx]===f[n.idx]:Q
})),d
}),d
},_space:function(){var a=[],c=this;
return P(S,function(d,e){c[e.cache]&&a.push(d)
}),a.pop()
},transition:function(d,p){var i=M(d),X=i._space(),u=S[X],c=0===this.alpha()?M("transparent"):this,l=c[u.cache]||u.to(c._rgba),f=l.slice();
return i=i[u.cache],P(u.props,function(r,Z){var s=Z.idx,h=l[s],e=i[s],Y=U[Z.type]||{};
null!==e&&(null===h?f[s]=e:(Y.mod&&(e-h>Y.mod/2?h+=Y.mod:h-e>Y.mod/2&&(h-=Y.mod)),f[s]=N((e-h)*p+h,Z)))
}),this[X](f)
},blend:function(d){if(1===this._rgba[3]){return this
}var a=this._rgba.slice(),c=a.pop(),f=M(d)._rgba;
return M(V.map(a,function(h,i){return(1-c)*f[i]+c*h
}))
},toRgbaString:function(){var c="rgba(",a=V.map(this._rgba,function(d,f){return null==d?f>2?1:0:d
});
return 1===a[3]&&(a.pop(),c="rgb("),c+a.join()+")"
},toHslaString:function(){var c="hsla(",a=V.map(this.hsla(),function(d,f){return null==d&&(d=f>2?1:0),f&&3>f&&(d=Math.round(100*d)+"%"),d
});
return 1===a[3]&&(a.pop(),c="hsl("),c+a.join()+")"
},toHexString:function(d){var a=this._rgba.slice(),c=a.pop();
return d&&a.push(~~(255*c)),"#"+V.map(a,function(e){return e=(e||0).toString(16),1===e.length?"0"+e:e
}).join("")
},toString:function(){return 0===this._rgba[3]?"transparent":this.toRgbaString()
}}),M.fn.parse.prototype=M.fn,S.hsla.to=function(ae){if(null==ae[0]||null==ae[1]||null==ae[2]){return[null,null,null,ae[3]]
}var aa,Y,af=ae[0]/255,p=ae[1]/255,f=ae[2]/255,ac=ae[3],d=Math.max(af,p,f),Z=Math.min(af,p,f),X=d-Z,ab=d+Z,ad=0.5*ab;
return aa=Z===d?0:af===d?60*(p-f)/X+360:p===d?60*(f-af)/X+120:60*(af-p)/X+240,Y=0===X?0:0.5>=ad?X/ab:X/(2-ab),[Math.round(aa)%360,Y,ad,null==ac?1:ac]
},S.hsla.from=function(f){if(null==f[0]||null==f[1]||null==f[2]){return[null,null,null,f[3]]
}var n=f[0]/360,d=f[1],h=f[2],p=f[3],c=0.5>=h?h*(1+d):h+d-h*d,l=2*h-c;
return[Math.round(255*L(l,c,n+1/3)),Math.round(255*L(l,c,n)),Math.round(255*L(l,c,n-1/3)),p]
},P(S,function(f,p){var i=p.props,d=p.cache,e=p.to,l=p.from;
M.fn[f]=function(c){if(e&&!this[d]&&(this[d]=e(this._rgba)),c===Q){return this[d].slice()
}var t,h=V.type(c),a="array"===h||"object"===h?c:arguments,o=this[d].slice();
return P(i,function(n,u){var r=a["object"===h?n:u.idx];
null==r&&(r=o[u.idx]),o[u.idx]=N(r,u)
}),l?(t=M(l(o)),t[d]=o,t):M(o)
},P(i,function(c,a){M.fn[c]||(M.fn[c]=function(Y){var u,s=V.type(Y),t="alpha"===c?this._hsla?"hsla":"rgba":f,r=this[t](),X=r[a.idx];
return"undefined"===s?X:("function"===s&&(Y=Y.call(this,X),s=V.type(Y)),null==Y&&a.empty?this:("string"===s&&(u=g.exec(Y),u&&(Y=X+parseFloat(u[2])*("+"===u[1]?1:-1))),r[a.idx]=Y,this[t](r)))
})
})
}),M.hook=function(c){var a=c.split(" ");
P(a,function(f,d){V.cssHooks[d]={set:function(s,u){var t,i,p="";
if("transparent"!==u&&("string"!==V.type(u)||(t=W(u)))){if(u=M(t||u),!R.rgba&&1!==u._rgba[3]){for(i="backgroundColor"===d?s.parentNode:s;
(""===p||"transparent"===p)&&i&&i.style;
){try{p=V.css(i,"backgroundColor"),i=i.parentNode
}catch(l){}}u=u.blend(p&&"transparent"!==p?p:"_default")
}u=u.toRgbaString()
}try{s.style[d]=u
}catch(l){}}},V.fx.step[d]=function(h){h.colorInit||(h.start=M(h.elem,d),h.end=M(h.end),h.colorInit=!0),V.cssHooks[d].set(h.elem,h.start.transition(h.end,h.pos))
}
})
},M.hook(T),V.cssHooks.borderColor={expand:function(a){var c={};
return P(["Top","Right","Bottom","Left"],function(d,e){c["border"+e+"Color"]=a
}),c
}},v=V.Color.names={aqua:"#00ffff",black:"#000000",blue:"#0000ff",fuchsia:"#ff00ff",gray:"#808080",green:"#008000",lime:"#00ff00",maroon:"#800000",navy:"#000080",olive:"#808000",purple:"#800080",red:"#ff0000",silver:"#c0c0c0",teal:"#008080",white:"#ffffff",yellow:"#ffff00",transparent:[null,null,null,0],_default:"#ffffff"}
}(x),function(){function d(l){var g,h,p=l.ownerDocument.defaultView?l.ownerDocument.defaultView.getComputedStyle(l,null):l.currentStyle,m={};
if(p&&p.length&&p[0]&&p[p[0]]){for(h=p.length;
h--;
){g=p[h],"string"==typeof p[g]&&(m[k.camelCase(g)]=p[g])
}}else{for(g in p){"string"==typeof p[g]&&(m[g]=p[g])
}}return m
}function a(m,h){var l,n,g={};
for(l in h){n=h[l],m[l]!==n&&(f[l]||(k.fx.step[l]||!isNaN(parseFloat(n)))&&(g[l]=n))
}return g
}var c=["add","remove","toggle"],f={border:1,borderBottom:1,borderColor:1,borderLeft:1,borderRight:1,borderTop:1,borderWidth:1,margin:1,padding:1};
k.each(["borderLeftStyle","borderRightStyle","borderBottomStyle","borderTopStyle"],function(h,g){k.fx.step[g]=function(e){("none"!==e.end&&!e.setAttr||1===e.pos&&!e.setAttr)&&(x.style(e.elem,g,e.end),e.setAttr=!0)
}
}),k.fn.addBack||(k.fn.addBack=function(e){return this.add(null==e?this.prevObject:this.prevObject.filter(e))
}),k.effects.animateClass=function(m,l,e,i){var g=k.speed(l,e,i);
return this.queue(function(){var s,n=k(this),p=n.attr("class")||"",h=g.children?n.find("*").addBack():n;
h=h.map(function(){var o=k(this);
return{el:o,start:d(this)}
}),s=function(){k.each(c,function(o,r){m[r]&&n[r+"Class"](m[r])
})
},s(),h=h.map(function(){return this.end=d(this.el[0]),this.diff=a(this.start,this.end),this
}),n.attr("class",p),h=h.map(function(){var t=this,o=k.Deferred(),r=k.extend({},g,{queue:!1,complete:function(){o.resolve(t)
}});
return this.el.animate(this.diff,r),o.promise()
}),k.when.apply(k,h.get()).done(function(){s(),k.each(arguments,function(){var o=this.el;
k.each(this.diff,function(r){o.css(r,"")
})
}),g.complete.call(n[0])
})
})
},k.fn.extend({addClass:function(g){return function(e,h,m,l){return h?k.effects.animateClass.call(this,{add:e},h,m,l):g.apply(this,arguments)
}
}(k.fn.addClass),removeClass:function(g){return function(e,h,m,l){return arguments.length>1?k.effects.animateClass.call(this,{remove:e},h,m,l):g.apply(this,arguments)
}
}(k.fn.removeClass),toggleClass:function(g){return function(h,l,p,m,e){return"boolean"==typeof l||void 0===l?p?k.effects.animateClass.call(this,l?{add:h}:{remove:h},p,m,e):g.apply(this,arguments):k.effects.animateClass.call(this,{toggle:h},l,p,m)
}
}(k.fn.toggleClass),switchClass:function(l,g,h,p,m){return k.effects.animateClass.call(this,{add:g,remove:l},h,p,m)
}})
}(),function(){function d(h,f,g,l){return k.isPlainObject(h)&&(f=h,h=h.effect),h={effect:h},null==f&&(f={}),k.isFunction(f)&&(l=f,g=null,f={}),("number"==typeof f||k.fx.speeds[f])&&(l=g,g=f,f={}),k.isFunction(g)&&(l=g,g=null),f&&k.extend(h,f),g=g||f.duration,h.duration=k.fx.off?0:"number"==typeof g?g:g in k.fx.speeds?k.fx.speeds[g]:k.fx.speeds._default,h.complete=l||f.complete,h
}function a(f){return !f||"number"==typeof f||k.fx.speeds[f]?!0:"string"!=typeof f||k.effects.effect[f]?k.isFunction(f)?!0:"object"!=typeof f||f.effect?!1:!0:!0
}function c(g,l){var f=l.outerWidth(),h=l.outerHeight(),p=/^rect\((-?\d*\.?\d*px|-?\d+%|auto),?\s*(-?\d*\.?\d*px|-?\d+%|auto),?\s*(-?\d*\.?\d*px|-?\d+%|auto),?\s*(-?\d*\.?\d*px|-?\d+%|auto)\)$/,m=p.exec(g)||["",0,f,h,0];
return{top:parseFloat(m[1])||0,right:"auto"===m[2]?f:parseFloat(m[2]),bottom:"auto"===m[3]?h:parseFloat(m[3]),left:parseFloat(m[4])||0}
}k.expr&&k.expr.filters&&k.expr.filters.animated&&(k.expr.filters.animated=function(f){return function(e){return !!k(e).data(H)||f(e)
}
}(k.expr.filters.animated)),k.uiBackCompat!==!1&&k.extend(k.effects,{save:function(g,l){for(var f=0,h=l.length;
h>f;
f++){null!==l[f]&&g.data(I+l[f],g[0].style[l[f]])
}},restore:function(g,l){for(var f,h=0,m=l.length;
m>h;
h++){null!==l[h]&&(f=g.data(I+l[h]),g.css(l[h],f))
}},setMode:function(f,g){return"toggle"===g&&(g=f.is(":hidden")?"show":"hide"),g
},createWrapper:function(l){if(l.parent().is(".ui-effects-wrapper")){return l.parent()
}var g={width:l.outerWidth(!0),height:l.outerHeight(!0),"float":l.css("float")},h=k("<div></div>").addClass("ui-effects-wrapper").css({fontSize:"100%",background:"transparent",border:"none",margin:0,padding:0}),p={width:l.width(),height:l.height()},m=document.activeElement;
try{m.id
}catch(f){m=document.body
}return l.wrap(h),(l[0]===m||k.contains(l[0],m))&&k(m).trigger("focus"),h=l.parent(),"static"===l.css("position")?(h.css({position:"relative"}),l.css({position:"relative"})):(k.extend(g,{position:l.css("position"),zIndex:l.css("z-index")}),k.each(["top","left","bottom","right"],function(e,i){g[i]=l.css(i),isNaN(parseInt(g[i],10))&&(g[i]="auto")
}),l.css({position:"relative",top:0,left:0,right:"auto",bottom:"auto"})),l.css(p),h.css(g).show()
},removeWrapper:function(g){var f=document.activeElement;
return g.parent().is(".ui-effects-wrapper")&&(g.parent().replaceWith(g),(g[0]===f||k.contains(g[0],f))&&k(f).trigger("focus")),g
}}),k.extend(k.effects,{version:"1.12.1",define:function(h,f,g){return g||(g=f,f="effect"),k.effects.effect[h]=g,k.effects.effect[h].mode=f,g
},scaledDimensions:function(g,l,f){if(0===l){return{height:0,width:0,outerHeight:0,outerWidth:0}
}var h="horizontal"!==f?(l||100)/100:1,m="vertical"!==f?(l||100)/100:1;
return{height:g.height()*m,width:g.width()*h,outerHeight:g.outerHeight()*m,outerWidth:g.outerWidth()*h}
},clipToBox:function(e){return{width:e.clip.right-e.clip.left,height:e.clip.bottom-e.clip.top,left:e.clip.left,top:e.clip.top}
},unshift:function(g,l,f){var h=g.queue();
l>1&&h.splice.apply(h,[1,0].concat(h.splice(l,f))),g.dequeue()
},saveStyle:function(e){e.data(j,e[0].style.cssText)
},restoreStyle:function(e){e[0].style.cssText=e.data(j)||"",e.removeData(j)
},mode:function(g,h){var f=g.is(":hidden");
return"toggle"===h&&(h=f?"show":"hide"),(f?"hide"===h:"show"===h)&&(h="none"),h
},getBaseline:function(g,l){var f,h;
switch(g[0]){case"top":f=0;
break;
case"middle":f=0.5;
break;
case"bottom":f=1;
break;
default:f=g[0]/l.height
}switch(g[1]){case"left":h=0;
break;
case"center":h=0.5;
break;
case"right":h=1;
break;
default:h=g[1]/l.width
}return{x:h,y:f}
},createPlaceholder:function(h){var f,g=h.css("position"),l=h.position();
return h.css({marginTop:h.css("marginTop"),marginBottom:h.css("marginBottom"),marginLeft:h.css("marginLeft"),marginRight:h.css("marginRight")}).outerWidth(h.outerWidth()).outerHeight(h.outerHeight()),/^(static|relative)/.test(g)&&(g="absolute",f=k("<"+h[0].nodeName+">").insertAfter(h).css({display:/^(inline|ruby)/.test(h.css("display"))?"inline-block":"block",visibility:"hidden",marginTop:h.css("marginTop"),marginBottom:h.css("marginBottom"),marginLeft:h.css("marginLeft"),marginRight:h.css("marginRight"),"float":h.css("float")}).outerWidth(h.outerWidth()).outerHeight(h.outerHeight()).addClass("ui-effects-placeholder"),h.data(I+"placeholder",f)),h.css({position:g,left:l.left,top:l.top}),f
},removePlaceholder:function(g){var h=I+"placeholder",f=g.data(h);
f&&(f.remove(),g.removeData(h))
},cleanUp:function(f){k.effects.restoreStyle(f),k.effects.removePlaceholder(f)
},setTransition:function(h,f,g,l){return l=l||{},k.each(f,function(m,e){var n=h.cssUnit(e);
n[0]>0&&(l[e]=n[0]*g+n[1])
}),l
}}),k.fn.extend({effect:function(){function p(o){function l(){n.removeData(H),k.effects.cleanUp(n),"hide"===N.mode&&n.hide(),h()
}function h(){k.isFunction(t)&&t.call(n[0]),k.isFunction(o)&&o()
}var n=k(this);
N.mode=v.shift(),k.uiBackCompat===!1||f?"none"===N.mode?(n[m](),h()):g.call(n[0],N,l):(n.is(":hidden")?"hide"===m:"show"===m)?(n[m](),h()):g.call(n[0],N,h)
}var N=d.apply(this,arguments),g=k.effects.effect[N.effect],f=g.mode,L=N.queue,e=L||"fx",t=N.complete,m=N.mode,v=[],M=function(n){var h=k(this),l=k.effects.mode(h,m)||f;
h.data(H,!0),v.push(l),f&&("show"===l||l===f&&"hide"===l)&&h.show(),f&&"none"===l||k.effects.saveStyle(h),k.isFunction(n)&&n()
};
return k.fx.off||!g?m?this[m](N.duration,t):this.each(function(){t&&t.call(this)
}):L===!1?this.each(M).each(p):this.queue(e,M).queue(e,p)
},show:function(e){return function(f){if(a(f)){return e.apply(this,arguments)
}var g=d.apply(this,arguments);
return g.mode="show",this.effect.call(this,g)
}
}(k.fn.show),hide:function(e){return function(f){if(a(f)){return e.apply(this,arguments)
}var g=d.apply(this,arguments);
return g.mode="hide",this.effect.call(this,g)
}
}(k.fn.hide),toggle:function(e){return function(f){if(a(f)||"boolean"==typeof f){return e.apply(this,arguments)
}var g=d.apply(this,arguments);
return g.mode="toggle",this.effect.call(this,g)
}
}(k.fn.toggle),cssUnit:function(h){var f=this.css(h),g=[];
return k.each(["em","px","%","pt"],function(i,l){f.indexOf(l)>0&&(g=[parseFloat(f),l])
}),g
},cssClip:function(e){return e?this.css("clip","rect("+e.top+"px "+e.right+"px "+e.bottom+"px "+e.left+"px)"):c(this.css("clip"),this)
},transfer:function(L,t){var Q=k(this),m=k(L.to),g="fixed"===m.css("position"),O=k("body"),f=g?O.scrollTop():0,v=g?O.scrollLeft():0,p=m.offset(),N={top:p.top-f,left:p.left-v,height:m.innerHeight(),width:m.innerWidth()},P=Q.offset(),M=k("<div class='ui-effects-transfer'></div>").appendTo("body").addClass(L.className).css({top:P.top-f,left:P.left-v,height:Q.innerHeight(),width:Q.innerWidth(),position:g?"fixed":"absolute"}).animate(N,L.duration,L.easing,function(){M.remove(),k.isFunction(t)&&t()
})
}}),k.fx.step.clip=function(f){f.clipInit||(f.start=k(f.elem).cssClip(),"string"==typeof f.end&&(f.end=c(f.end,f.elem)),f.clipInit=!0),k(f.elem).cssClip({top:f.pos*(f.end.top-f.start.top)+f.start.top,right:f.pos*(f.end.right-f.start.right)+f.start.right,bottom:f.pos*(f.end.bottom-f.start.bottom)+f.start.bottom,left:f.pos*(f.end.left-f.start.left)+f.start.left})
}
}(),function(){var a={};
k.each(["Quad","Cubic","Quart","Quint","Expo"],function(d,c){a[c]=function(f){return Math.pow(f,d+2)
}
}),k.extend(a,{Sine:function(c){return 1-Math.cos(c*Math.PI/2)
},Circ:function(c){return 1-Math.sqrt(1-c*c)
},Elastic:function(c){return 0===c||1===c?c:-Math.pow(2,8*(c-1))*Math.sin((80*(c-1)-7.5)*Math.PI/15)
},Back:function(c){return c*c*(3*c-2)
},Bounce:function(d){for(var f,c=4;
((f=Math.pow(2,--c))-1)/11>d;
){}return 1/Math.pow(4,3-c)-7.5625*Math.pow((3*f-2)/22-d,2)
}}),k.each(a,function(d,c){k.easing["easeIn"+d]=c,k.easing["easeOut"+d]=function(e){return 1-c(1-e)
},k.easing["easeInOut"+d]=function(e){return 0.5>e?c(2*e)/2:1-c(-2*e+2)/2
}
})
}();
var F=k.effects;
k.effects.define("blind","hide",function(m,d){var g={up:["bottom","top"],vertical:["bottom","top"],down:["top","bottom"],left:["right","left"],horizontal:["right","left"],right:["left","right"]},t=k(this),p=m.direction||"up",c=t.cssClip(),l={clip:k.extend({},c)},f=k.effects.createPlaceholder(t);
l.clip[g[p][0]]=l.clip[g[p][1]],"show"===m.mode&&(t.cssClip(l.clip),f&&f.css(k.effects.clipToBox(l)),l.clip=c),f&&f.animate(k.effects.clipToBox(l),m.duration,m.easing),t.animate(l,{queue:!1,duration:m.duration,easing:m.easing,complete:d})
}),k.effects.define("bounce",function(X,T){var M,Q,P,ab=k(this),N=X.mode,U="hide"===N,S="show"===N,Z=X.direction||"up",L=X.distance,Y=X.times||5,O=2*Y+(S||U?1:0),W=X.duration/O,V=X.easing,R="up"===Z||"down"===Z?"top":"left",ac="up"===Z||"left"===Z,t=0,aa=ab.queue().length;
for(k.effects.createPlaceholder(ab),P=ab.css(R),L||(L=ab["top"===R?"outerHeight":"outerWidth"]()/3),S&&(Q={opacity:1},Q[R]=P,ab.css("opacity",0).css(R,ac?2*-L:2*L).animate(Q,W,V)),U&&(L/=Math.pow(2,Y-1)),Q={},Q[R]=P;
Y>t;
t++){M={},M[R]=(ac?"-=":"+=")+L,ab.animate(M,W,V).animate(Q,W,V),L=U?2*L:L/2
}U&&(M={opacity:0},M[R]=(ac?"-=":"+=")+L,ab.animate(M,W,V)),ab.queue(T),k.effects.unshift(ab,aa,O+1)
}),k.effects.define("clip","hide",function(t,m){var v,f={},d=k(this),u=t.direction||"vertical",c="both"===u,p=c||"horizontal"===u,g=c||"vertical"===u;
v=d.cssClip(),f.clip={top:g?(v.bottom-v.top)/2:v.top,right:p?(v.right-v.left)/2:v.right,bottom:g?(v.bottom-v.top)/2:v.bottom,left:p?(v.right-v.left)/2:v.left},k.effects.createPlaceholder(d),"show"===t.mode&&(d.cssClip(f.clip),f.clip=v),d.animate(f,{queue:!1,duration:t.duration,easing:t.easing,complete:m})
}),k.effects.define("drop","hide",function(v,p){var O,g=k(this),f=v.mode,M="show"===f,d=v.direction||"left",t="up"===d||"down"===d?"top":"left",m="up"===d||"left"===d?"-=":"+=",L="+="===m?"-=":"+=",N={opacity:0};
k.effects.createPlaceholder(g),O=v.distance||g["top"===t?"outerHeight":"outerWidth"](!0)/2,N[t]=m+O,M&&(g.css(N),N[t]=L+O,N.opacity=1),g.animate(N,{queue:!1,duration:v.duration,easing:v.easing,complete:p})
}),k.effects.define("explode","hide",function(X,T){function M(){aa.push(this),aa.length===L*Y&&Q()
}function Q(){O.css({visibility:"visible"}),k(aa).remove(),T()
}var P,ab,N,U,S,Z,L=X.pieces?Math.round(Math.sqrt(X.pieces)):3,Y=L,O=k(this),W=X.mode,V="show"===W,R=O.show().css("visibility","hidden").offset(),ac=Math.ceil(O.outerWidth()/Y),t=Math.ceil(O.outerHeight()/L),aa=[];
for(P=0;
L>P;
P++){for(U=R.top+P*t,Z=P-(L-1)/2,ab=0;
Y>ab;
ab++){N=R.left+ab*ac,S=ab-(Y-1)/2,O.clone().appendTo("body").wrap("<div></div>").css({position:"absolute",visibility:"visible",left:-ab*ac,top:-P*t}).parent().addClass("ui-effects-explode").css({position:"absolute",overflow:"hidden",width:ac,height:t,left:N+(V?S*ac:0),top:U+(V?Z*t:0),opacity:V?0:1}).animate({left:N+(V?0:S*ac),top:U+(V?0:Z*t),opacity:V?1:0},X.duration||500,X.easing,M)
}}}),k.effects.define("fade","toggle",function(d,a){var c="show"===d.mode;
k(this).css("opacity",c?0:1).animate({opacity:c?1:0},{queue:!1,duration:d.duration,easing:d.easing,complete:a})
}),k.effects.define("fold","hide",function(T,P){var Z=k(this),M=T.mode,L="show"===M,W="hide"===M,t=T.size||15,Q=/([0-9]+)%/.exec(t),O=!!T.horizFirst,V=O?["right","bottom"]:["bottom","right"],Y=T.duration/2,U=k.effects.createPlaceholder(Z),v=Z.cssClip(),S={clip:k.extend({},v)},R={clip:k.extend({},v)},N=[v[V[0]],v[V[1]]],X=Z.queue().length;
Q&&(t=parseInt(Q[1],10)/100*N[W?0:1]),S.clip[V[0]]=t,R.clip[V[0]]=t,R.clip[V[1]]=0,L&&(Z.cssClip(R.clip),U&&U.css(k.effects.clipToBox(R)),R.clip=v),Z.queue(function(a){U&&U.animate(k.effects.clipToBox(S),Y,T.easing).animate(k.effects.clipToBox(R),Y,T.easing),a()
}).animate(S,Y,T.easing).animate(R,Y,T.easing).queue(P),k.effects.unshift(Z,X,4)
}),k.effects.define("highlight","show",function(d,a){var c=k(this),f={backgroundColor:c.css("backgroundColor")};
"hide"===d.mode&&(f.opacity=0),k.effects.saveStyle(c),c.css({backgroundImage:"none",backgroundColor:d.color||"#ffff99"}).animate(f,{queue:!1,duration:d.duration,easing:d.easing,complete:a})
}),k.effects.define("size",function(X,T){var M,Q,P,aa=k(this),N=["fontSize"],U=["borderTopWidth","borderBottomWidth","paddingTop","paddingBottom"],S=["borderLeftWidth","borderRightWidth","paddingLeft","paddingRight"],Z=X.mode,L="effect"!==Z,Y=X.scale||"both",O=X.origin||["middle","center"],W=aa.css("position"),V=aa.position(),R=k.effects.scaledDimensions(aa),ab=X.from||R,t=X.to||k.effects.scaledDimensions(aa,0);
k.effects.createPlaceholder(aa),"show"===Z&&(P=ab,ab=t,t=P),Q={from:{y:ab.height/R.height,x:ab.width/R.width},to:{y:t.height/R.height,x:t.width/R.width}},("box"===Y||"both"===Y)&&(Q.from.y!==Q.to.y&&(ab=k.effects.setTransition(aa,U,Q.from.y,ab),t=k.effects.setTransition(aa,U,Q.to.y,t)),Q.from.x!==Q.to.x&&(ab=k.effects.setTransition(aa,S,Q.from.x,ab),t=k.effects.setTransition(aa,S,Q.to.x,t))),("content"===Y||"both"===Y)&&Q.from.y!==Q.to.y&&(ab=k.effects.setTransition(aa,N,Q.from.y,ab),t=k.effects.setTransition(aa,N,Q.to.y,t)),O&&(M=k.effects.getBaseline(O,R),ab.top=(R.outerHeight-ab.outerHeight)*M.y+V.top,ab.left=(R.outerWidth-ab.outerWidth)*M.x+V.left,t.top=(R.outerHeight-t.outerHeight)*M.y+V.top,t.left=(R.outerWidth-t.outerWidth)*M.x+V.left),aa.css(ab),("content"===Y||"both"===Y)&&(U=U.concat(["marginTop","marginBottom"]).concat(N),S=S.concat(["marginLeft","marginRight"]),aa.find("*[width]").each(function(){var d=k(this),e=k.effects.scaledDimensions(d),f={height:e.height*Q.from.y,width:e.width*Q.from.x,outerHeight:e.outerHeight*Q.from.y,outerWidth:e.outerWidth*Q.from.x},c={height:e.height*Q.to.y,width:e.width*Q.to.x,outerHeight:e.height*Q.to.y,outerWidth:e.width*Q.to.x};
Q.from.y!==Q.to.y&&(f=k.effects.setTransition(d,U,Q.from.y,f),c=k.effects.setTransition(d,U,Q.to.y,c)),Q.from.x!==Q.to.x&&(f=k.effects.setTransition(d,S,Q.from.x,f),c=k.effects.setTransition(d,S,Q.to.x,c)),L&&k.effects.saveStyle(d),d.css(f),d.animate(c,X.duration,X.easing,function(){L&&k.effects.restoreStyle(d)
})
})),aa.animate(t,{queue:!1,duration:X.duration,easing:X.easing,complete:function(){var a=aa.offset();
0===t.opacity&&aa.css("opacity",ab.opacity),L||(aa.css("position","static"===W?"relative":W).offset(a),k.effects.saveStyle(aa)),T()
}})
}),k.effects.define("scale",function(g,d){var f=k(this),l=g.mode,h=parseInt(g.percent,10)||(0===parseInt(g.percent,10)?0:"effect"!==l?0:100),c=k.extend(!0,{from:k.effects.scaledDimensions(f),to:k.effects.scaledDimensions(f,h,g.direction||"both"),origin:g.origin||["middle","center"]},g);
g.fade&&(c.from.opacity=1,c.to.opacity=0),k.effects.effect.size.call(this,c,d)
}),k.effects.define("puff","hide",function(d,a){var c=k.extend(!0,{},d,{fade:!0,percent:parseInt(d.percent,10)||150});
k.effects.effect.scale.call(this,c,a)
}),k.effects.define("pulsate","show",function(L,t){var Q=k(this),m=L.mode,g="show"===m,O="hide"===m,f=g||O,v=2*(L.times||5)+(f?1:0),p=L.duration/v,N=0,P=1,M=Q.queue().length;
for((g||!Q.is(":visible"))&&(Q.css("opacity",0).show(),N=1);
v>P;
P++){Q.animate({opacity:N},p,L.easing),N=1-N
}Q.animate({opacity:N},p,L.easing),Q.queue(t),k.effects.unshift(Q,M,v+1)
}),k.effects.define("shake",function(R,N){var W=1,L=k(this),v=R.direction||"left",U=R.distance||20,m=R.times||3,O=2*m+1,M=Math.round(R.duration/O),T="up"===v||"down"===v?"top":"left",V="up"===v||"left"===v,S={},t={},Q={},P=L.queue().length;
for(k.effects.createPlaceholder(L),S[T]=(V?"-=":"+=")+U,t[T]=(V?"+=":"-=")+2*U,Q[T]=(V?"-=":"+=")+2*U,L.animate(S,M,R.easing);
m>W;
W++){L.animate(t,M,R.easing).animate(Q,M,R.easing)
}L.animate(t,M,R.easing).animate(S,M/2,R.easing).queue(N),k.effects.unshift(L,P,O+1)
}),k.effects.define("slide","show",function(L,t){var Q,m,g=k(this),O={up:["bottom","top"],down:["top","bottom"],left:["right","left"],right:["left","right"]},f=L.mode,v=L.direction||"left",p="up"===v||"down"===v?"top":"left",N="up"===v||"left"===v,P=L.distance||g["top"===p?"outerHeight":"outerWidth"](!0),M={};
k.effects.createPlaceholder(g),Q=g.cssClip(),m=g.position()[p],M[p]=(N?-1:1)*P+m,M.clip=g.cssClip(),M.clip[O[v][1]]=M.clip[O[v][0]],"show"===f&&(g.cssClip(M.clip),g.css(p,M[p]),M.clip=Q,M[p]=m),g.animate(M,{queue:!1,duration:L.duration,easing:L.easing,complete:t})
});
var F;
k.uiBackCompat!==!1&&(F=k.effects.define("transfer",function(c,a){k(this).transfer(c,a)
})),k.ui.focusable=function(e,g){var t,p,d,m,f,c=e.nodeName.toLowerCase();
return"area"===c?(t=e.parentNode,p=t.name,e.href&&p&&"map"===t.nodeName.toLowerCase()?(d=k("img[usemap='#"+p+"']"),d.length>0&&d.is(":visible")):!1):(/^(input|select|textarea|button|object)$/.test(c)?(m=!e.disabled,m&&(f=k(e).closest("fieldset")[0],f&&(m=!f.disabled))):m="a"===c?e.href||g:g,m&&k(e).is(":visible")&&G(k(e)))
},k.extend(k.expr[":"],{focusable:function(a){return k.ui.focusable(a,null!=k.attr(a,"tabindex"))
}}),k.ui.focusable,k.fn.form=function(){return"string"==typeof this[0].form?this.closest("form"):k(this[0].form)
},k.ui.formResetMixin={_formResetHandler:function(){var a=k(this);
setTimeout(function(){var c=a.data("ui-form-reset-instances");
k.each(c,function(){this.refresh()
})
})
},_bindFormResetHandler:function(){if(this.form=this.element.form(),this.form.length){var a=this.form.data("ui-form-reset-instances")||[];
a.length||this.form.on("reset.ui-form-reset",this._formResetHandler),a.push(this),this.form.data("ui-form-reset-instances",a)
}},_unbindFormResetHandler:function(){if(this.form.length){var a=this.form.data("ui-form-reset-instances");
a.splice(k.inArray(this,a),1),a.length?this.form.data("ui-form-reset-instances",a):this.form.removeData("ui-form-reset-instances").off("reset.ui-form-reset")
}}},"1.7"===k.fn.jquery.substring(0,3)&&(k.each(["Width","Height"],function(g,d){function f(n,a,m,p){return k.each(l,function(){a-=parseFloat(k.css(n,"padding"+this))||0,m&&(a-=parseFloat(k.css(n,"border"+this+"Width"))||0),p&&(a-=parseFloat(k.css(n,"margin"+this))||0)
}),a
}var l="Width"===d?["Left","Right"]:["Top","Bottom"],h=d.toLowerCase(),c={innerWidth:k.fn.innerWidth,innerHeight:k.fn.innerHeight,outerWidth:k.fn.outerWidth,outerHeight:k.fn.outerHeight};
k.fn["inner"+d]=function(a){return void 0===a?c["inner"+d].call(this):this.each(function(){k(this).css(h,f(this,a)+"px")
})
},k.fn["outer"+d]=function(a,i){return"number"!=typeof a?c["outer"+d].call(this,a):this.each(function(){k(this).css(h,f(this,a,!0,i)+"px")
})
}
}),k.fn.addBack=function(a){return this.add(null==a?this.prevObject:this.prevObject.filter(a))
}),k.ui.keyCode={BACKSPACE:8,COMMA:188,DELETE:46,DOWN:40,END:35,ENTER:13,ESCAPE:27,HOME:36,LEFT:37,PAGE_DOWN:34,PAGE_UP:33,PERIOD:190,RIGHT:39,SPACE:32,TAB:9,UP:38},k.ui.escapeSelector=function(){var a=/([!"#$%&'()*+,.\/:;<=>?@[\]^`{|}~])/g;
return function(c){return c.replace(a,"\\$1")
}
}(),k.fn.labels=function(){var d,a,c,g,f;
return this[0].labels&&this[0].labels.length?this.pushStack(this[0].labels):(g=this.eq(0).parents("label"),c=this.attr("id"),c&&(d=this.eq(0).parents().last(),f=d.add(d.length?d.siblings():this.siblings()),a="label[for='"+k.ui.escapeSelector(c)+"']",g=g.add(f.find(a).addBack(a))),this.pushStack(g))
},k.fn.scrollParent=function(d){var a=this.css("position"),c="absolute"===a,g=d?/(auto|scroll|hidden)/:/(auto|scroll)/,f=this.parents().filter(function(){var h=k(this);
return c&&"static"===h.css("position")?!1:g.test(h.css("overflow")+h.css("overflow-y")+h.css("overflow-x"))
}).eq(0);
return"fixed"!==a&&f.length?f:k(this[0].ownerDocument||document)
},k.extend(k.expr[":"],{tabbable:function(d){var a=k.attr(d,"tabindex"),c=null!=a;
return(!c||a>=0)&&k.ui.focusable(d,c)
}}),k.fn.extend({uniqueId:function(){var a=0;
return function(){return this.each(function(){this.id||(this.id="ui-id-"+ ++a)
})
}
}(),removeUniqueId:function(){return this.each(function(){/^ui-id-\d+$/.test(this.id)&&k(this).removeAttr("id")
})
}}),k.widget("ui.accordion",{version:"1.12.1",options:{active:0,animate:{},classes:{"ui-accordion-header":"ui-corner-top","ui-accordion-header-collapsed":"ui-corner-all","ui-accordion-content":"ui-corner-bottom"},collapsible:!1,event:"click",header:"> li > :first-child, > :not(li):even",heightStyle:"auto",icons:{activeHeader:"ui-icon-triangle-1-s",header:"ui-icon-triangle-1-e"},activate:null,beforeActivate:null},hideProps:{borderTopWidth:"hide",borderBottomWidth:"hide",paddingTop:"hide",paddingBottom:"hide",height:"hide"},showProps:{borderTopWidth:"show",borderBottomWidth:"show",paddingTop:"show",paddingBottom:"show",height:"show"},_create:function(){var a=this.options;
this.prevShow=this.prevHide=k(),this._addClass("ui-accordion","ui-widget ui-helper-reset"),this.element.attr("role","tablist"),a.collapsible||a.active!==!1&&null!=a.active||(a.active=0),this._processPanels(),0>a.active&&(a.active+=this.headers.length),this._refresh()
},_getCreateEventData:function(){return{header:this.active,panel:this.active.length?this.active.next():k()}
},_createIcons:function(){var d,a,c=this.options.icons;
c&&(d=k("<span>"),this._addClass(d,"ui-accordion-header-icon","ui-icon "+c.header),d.prependTo(this.headers),a=this.active.children(".ui-accordion-header-icon"),this._removeClass(a,c.header)._addClass(a,null,c.activeHeader)._addClass(this.headers,"ui-accordion-icons"))
},_destroyIcons:function(){this._removeClass(this.headers,"ui-accordion-icons"),this.headers.children(".ui-accordion-header-icon").remove()
},_destroy:function(){var a;
this.element.removeAttr("role"),this.headers.removeAttr("role aria-expanded aria-selected aria-controls tabIndex").removeUniqueId(),this._destroyIcons(),a=this.headers.next().css("display","").removeAttr("role aria-hidden aria-labelledby").removeUniqueId(),"content"!==this.options.heightStyle&&a.css("height","")
},_setOption:function(a,c){return"active"===a?(this._activate(c),void 0):("event"===a&&(this.options.event&&this._off(this.headers,this.options.event),this._setupEvents(c)),this._super(a,c),"collapsible"!==a||c||this.options.active!==!1||this._activate(0),"icons"===a&&(this._destroyIcons(),c&&this._createIcons()),void 0)
},_setOptionDisabled:function(a){this._super(a),this.element.attr("aria-disabled",a),this._toggleClass(null,"ui-state-disabled",!!a),this._toggleClass(this.headers.add(this.headers.next()),null,"ui-state-disabled",!!a)
},_keydown:function(d){if(!d.altKey&&!d.ctrlKey){var a=k.ui.keyCode,c=this.headers.length,g=this.headers.index(d.target),f=!1;
switch(d.keyCode){case a.RIGHT:case a.DOWN:f=this.headers[(g+1)%c];
break;
case a.LEFT:case a.UP:f=this.headers[(g-1+c)%c];
break;
case a.SPACE:case a.ENTER:this._eventHandler(d);
break;
case a.HOME:f=this.headers[0];
break;
case a.END:f=this.headers[c-1]
}f&&(k(d.target).attr("tabIndex",-1),k(f).attr("tabIndex",0),k(f).trigger("focus"),d.preventDefault())
}},_panelKeyDown:function(a){a.keyCode===k.ui.keyCode.UP&&a.ctrlKey&&k(a.currentTarget).prev().trigger("focus")
},refresh:function(){var a=this.options;
this._processPanels(),a.active===!1&&a.collapsible===!0||!this.headers.length?(a.active=!1,this.active=k()):a.active===!1?this._activate(0):this.active.length&&!k.contains(this.element[0],this.active[0])?this.headers.length===this.headers.find(".ui-state-disabled").length?(a.active=!1,this.active=k()):this._activate(Math.max(0,a.active-1)):a.active=this.headers.index(this.active),this._destroyIcons(),this._refresh()
},_processPanels:function(){var a=this.headers,c=this.panels;
this.headers=this.element.find(this.options.header),this._addClass(this.headers,"ui-accordion-header ui-accordion-header-collapsed","ui-state-default"),this.panels=this.headers.next().filter(":not(.ui-accordion-content-active)").hide(),this._addClass(this.panels,"ui-accordion-content","ui-helper-reset ui-widget-content"),c&&(this._off(a.not(this.headers)),this._off(c.not(this.panels)))
},_refresh:function(){var d,a=this.options,c=a.heightStyle,f=this.element.parent();
this.active=this._findActive(a.active),this._addClass(this.active,"ui-accordion-header-active","ui-state-active")._removeClass(this.active,"ui-accordion-header-collapsed"),this._addClass(this.active.next(),"ui-accordion-content-active"),this.active.next().show(),this.headers.attr("role","tab").each(function(){var l=k(this),g=l.uniqueId().attr("id"),h=l.next(),m=h.uniqueId().attr("id");
l.attr("aria-controls",m),h.attr("aria-labelledby",g)
}).next().attr("role","tabpanel"),this.headers.not(this.active).attr({"aria-selected":"false","aria-expanded":"false",tabIndex:-1}).next().attr({"aria-hidden":"true"}).hide(),this.active.length?this.active.attr({"aria-selected":"true","aria-expanded":"true",tabIndex:0}).next().attr({"aria-hidden":"false"}):this.headers.eq(0).attr("tabIndex",0),this._createIcons(),this._setupEvents(a.event),"fill"===c?(d=f.height(),this.element.siblings(":visible").each(function(){var e=k(this),g=e.css("position");
"absolute"!==g&&"fixed"!==g&&(d-=e.outerHeight(!0))
}),this.headers.each(function(){d-=k(this).outerHeight(!0)
}),this.headers.next().each(function(){k(this).height(Math.max(0,d-k(this).innerHeight()+k(this).height()))
}).css("overflow","auto")):"auto"===c&&(d=0,this.headers.next().each(function(){var e=k(this).is(":visible");
e||k(this).show(),d=Math.max(d,k(this).css("height","").height()),e||k(this).hide()
}).height(d))
},_activate:function(c){var a=this._findActive(c)[0];
a!==this.active[0]&&(a=a||this.active[0],this._eventHandler({target:a,currentTarget:a,preventDefault:k.noop}))
},_findActive:function(a){return"number"==typeof a?this.headers.eq(a):k()
},_setupEvents:function(c){var a={keydown:"_keydown"};
c&&k.each(c.split(" "),function(d,f){a[f]="_eventHandler"
}),this._off(this.headers.add(this.headers.next())),this._on(this.headers,a),this._on(this.headers.next(),{keydown:"_panelKeyDown"}),this._hoverable(this.headers),this._focusable(this.headers)
},_eventHandler:function(v){var p,O,g=this.options,f=this.active,M=k(v.currentTarget),d=M[0]===f[0],t=d&&g.collapsible,m=t?k():M.next(),L=f.next(),N={oldHeader:f,oldPanel:L,newHeader:t?k():M,newPanel:m};
v.preventDefault(),d&&!g.collapsible||this._trigger("beforeActivate",v,N)===!1||(g.active=t?!1:this.headers.index(M),this.active=d?k():M,this._toggle(N),this._removeClass(f,"ui-accordion-header-active","ui-state-active"),g.icons&&(p=f.children(".ui-accordion-header-icon"),this._removeClass(p,null,g.icons.activeHeader)._addClass(p,null,g.icons.header)),d||(this._removeClass(M,"ui-accordion-header-collapsed")._addClass(M,"ui-accordion-header-active","ui-state-active"),g.icons&&(O=M.children(".ui-accordion-header-icon"),this._removeClass(O,null,g.icons.header)._addClass(O,null,g.icons.activeHeader)),this._addClass(M.next(),"ui-accordion-content-active")))
},_toggle:function(d){var a=d.newPanel,c=this.prevShow.length?this.prevShow:d.oldPanel;
this.prevShow.add(this.prevHide).stop(!0,!0),this.prevShow=a,this.prevHide=c,this.options.animate?this._animate(a,c,d):(c.hide(),a.show(),this._toggleComplete(d)),c.attr({"aria-hidden":"true"}),c.prev().attr({"aria-selected":"false","aria-expanded":"false"}),a.length&&c.length?c.prev().attr({tabIndex:-1,"aria-expanded":"false"}):a.length&&this.headers.filter(function(){return 0===parseInt(k(this).attr("tabIndex"),10)
}).attr("tabIndex",-1),a.attr("aria-hidden","false").prev().attr({"aria-selected":"true","aria-expanded":"true",tabIndex:0})
},_animate:function(R,M,v){var S,m,g,P=this,f=0,L=R.css("box-sizing"),p=R.length&&(!M.length||R.index()<M.index()),O=this.options.animate||{},Q=p&&O.down||O,N=function(){P._toggleComplete(v)
};
return"number"==typeof Q&&(g=Q),"string"==typeof Q&&(m=Q),m=m||Q.easing||O.easing,g=g||Q.duration||O.duration,M.length?R.length?(S=R.show().outerHeight(),M.animate(this.hideProps,{duration:g,easing:m,step:function(a,c){c.now=Math.round(a)
}}),R.hide().animate(this.showProps,{duration:g,easing:m,complete:N,step:function(c,a){a.now=Math.round(c),"height"!==a.prop?"content-box"===L&&(f+=a.now):"content"!==P.options.heightStyle&&(a.now=Math.round(S-M.outerHeight()-f),f=0)
}}),void 0):M.animate(this.hideProps,g,m,N):R.animate(this.showProps,g,m,N)
},_toggleComplete:function(c){var d=c.oldPanel,a=d.prev();
this._removeClass(d,"ui-accordion-content-active"),this._removeClass(a,"ui-accordion-header-active")._addClass(a,"ui-accordion-header-collapsed"),d.length&&(d.parent()[0].className=d.parent()[0].className),this._trigger("activate",null,c)
}}),k.ui.safeActiveElement=function(c){var d;
try{d=c.activeElement
}catch(a){d=c.body
}return d||(d=c.body),d.nodeName||(d=c.body),d
},k.widget("ui.menu",{version:"1.12.1",defaultElement:"<ul>",delay:300,options:{icons:{submenu:"ui-icon-caret-1-e"},items:"> *",menus:"ul",position:{my:"left top",at:"right top"},role:"menu",blur:null,focus:null,select:null},_create:function(){this.activeMenu=this.element,this.mouseHandled=!1,this.element.uniqueId().attr({role:this.options.role,tabIndex:0}),this._addClass("ui-menu","ui-widget ui-widget-content"),this._on({"mousedown .ui-menu-item":function(a){a.preventDefault()
},"click .ui-menu-item":function(d){var a=k(d.target),c=k(k.ui.safeActiveElement(this.document[0]));
!this.mouseHandled&&a.not(".ui-state-disabled").length&&(this.select(d),d.isPropagationStopped()||(this.mouseHandled=!0),a.has(".ui-menu").length?this.expand(d):!this.element.is(":focus")&&c.closest(".ui-menu").length&&(this.element.trigger("focus",[!0]),this.active&&1===this.active.parents(".ui-menu").length&&clearTimeout(this.timer)))
},"mouseenter .ui-menu-item":function(d){if(!this.previousFilter){var a=k(d.target).closest(".ui-menu-item"),c=k(d.currentTarget);
a[0]===c[0]&&(this._removeClass(c.siblings().children(".ui-state-active"),null,"ui-state-active"),this.focus(d,c))
}},mouseleave:"collapseAll","mouseleave .ui-menu":"collapseAll",focus:function(c,d){var a=this.active||this.element.find(this.options.items).eq(0);
d||this.focus(c,a)
},blur:function(a){this._delay(function(){var c=!k.contains(this.element[0],k.ui.safeActiveElement(this.document[0]));
c&&this.collapseAll(a)
})
},keydown:"_keydown"}),this.refresh(),this._on(this.document,{click:function(a){this._closeOnDocumentClick(a)&&this.collapseAll(a),this.mouseHandled=!1
}})
},_destroy:function(){var c=this.element.find(".ui-menu-item").removeAttr("role aria-disabled"),a=c.children(".ui-menu-item-wrapper").removeUniqueId().removeAttr("tabIndex role aria-haspopup");
this.element.removeAttr("aria-activedescendant").find(".ui-menu").addBack().removeAttr("role aria-labelledby aria-expanded aria-hidden aria-disabled tabIndex").removeUniqueId().show(),a.children().each(function(){var d=k(this);
d.data("ui-menu-submenu-caret")&&d.remove()
})
},_keydown:function(g){var d,f,l,h,c=!0;
switch(g.keyCode){case k.ui.keyCode.PAGE_UP:this.previousPage(g);
break;
case k.ui.keyCode.PAGE_DOWN:this.nextPage(g);
break;
case k.ui.keyCode.HOME:this._move("first","first",g);
break;
case k.ui.keyCode.END:this._move("last","last",g);
break;
case k.ui.keyCode.UP:this.previous(g);
break;
case k.ui.keyCode.DOWN:this.next(g);
break;
case k.ui.keyCode.LEFT:this.collapse(g);
break;
case k.ui.keyCode.RIGHT:this.active&&!this.active.is(".ui-state-disabled")&&this.expand(g);
break;
case k.ui.keyCode.ENTER:case k.ui.keyCode.SPACE:this._activate(g);
break;
case k.ui.keyCode.ESCAPE:this.collapse(g);
break;
default:c=!1,f=this.previousFilter||"",h=!1,l=g.keyCode>=96&&105>=g.keyCode?""+(g.keyCode-96):String.fromCharCode(g.keyCode),clearTimeout(this.filterTimer),l===f?h=!0:l=f+l,d=this._filterMenuItems(l),d=h&&-1!==d.index(this.active.next())?this.active.nextAll(".ui-menu-item"):d,d.length||(l=String.fromCharCode(g.keyCode),d=this._filterMenuItems(l)),d.length?(this.focus(g,d),this.previousFilter=l,this.filterTimer=this._delay(function(){delete this.previousFilter
},1000)):delete this.previousFilter
}c&&g.preventDefault()
},_activate:function(a){this.active&&!this.active.is(".ui-state-disabled")&&(this.active.children("[aria-haspopup='true']").length?this.expand(a):this.select(a))
},refresh:function(){var m,d,g,t,p,c=this,l=this.options.icons.submenu,f=this.element.find(this.options.menus);
this._toggleClass("ui-menu-icons",null,!!this.element.find(".ui-icon").length),g=f.filter(":not(.ui-menu)").hide().attr({role:this.options.role,"aria-hidden":"true","aria-expanded":"false"}).each(function(){var n=k(this),a=n.prev(),h=k("<span>").data("ui-menu-submenu-caret",!0);
c._addClass(h,"ui-menu-icon","ui-icon "+l),a.attr("aria-haspopup","true").prepend(h),n.attr("aria-labelledby",a.attr("id"))
}),this._addClass(g,"ui-menu","ui-widget ui-widget-content ui-front"),m=f.add(this.element),d=m.find(this.options.items),d.not(".ui-menu-item").each(function(){var a=k(this);
c._isDivider(a)&&c._addClass(a,"ui-menu-divider","ui-widget-content")
}),t=d.not(".ui-menu-item, .ui-menu-divider"),p=t.children().not(".ui-menu").uniqueId().attr({tabIndex:-1,role:this._itemRole()}),this._addClass(t,"ui-menu-item")._addClass(p,"ui-menu-item-wrapper"),d.filter(".ui-state-disabled").attr("aria-disabled","true"),this.active&&!k.contains(this.element[0],this.active[0])&&this.blur()
},_itemRole:function(){return{menu:"menuitem",listbox:"option"}[this.options.role]
},_setOption:function(c,d){if("icons"===c){var a=this.element.find(".ui-menu-icon");
this._removeClass(a,null,this.options.icons.submenu)._addClass(a,null,d.submenu)
}this._super(c,d)
},_setOptionDisabled:function(a){this._super(a),this.element.attr("aria-disabled",a+""),this._toggleClass(null,"ui-state-disabled",!!a)
},focus:function(c,f){var a,d,g;
this.blur(c,c&&"focus"===c.type),this._scrollIntoView(f),this.active=f.first(),d=this.active.children(".ui-menu-item-wrapper"),this._addClass(d,null,"ui-state-active"),this.options.role&&this.element.attr("aria-activedescendant",d.attr("id")),g=this.active.parent().closest(".ui-menu-item").children(".ui-menu-item-wrapper"),this._addClass(g,null,"ui-state-active"),c&&"keydown"===c.type?this._close():this.timer=this._delay(function(){this._close()
},this.delay),a=f.children(".ui-menu"),a.length&&c&&/^mouse/.test(c.type)&&this._startOpening(a),this.activeMenu=f.parent(),this._trigger("focus",c,{item:f})
},_scrollIntoView:function(h){var d,f,m,l,c,g;
this._hasScroll()&&(d=parseFloat(k.css(this.activeMenu[0],"borderTopWidth"))||0,f=parseFloat(k.css(this.activeMenu[0],"paddingTop"))||0,m=h.offset().top-this.activeMenu.offset().top-d-f,l=this.activeMenu.scrollTop(),c=this.activeMenu.height(),g=h.outerHeight(),0>m?this.activeMenu.scrollTop(l+m):m+g>c&&this.activeMenu.scrollTop(l+m-c+g))
},blur:function(a,c){c||clearTimeout(this.timer),this.active&&(this._removeClass(this.active.children(".ui-menu-item-wrapper"),null,"ui-state-active"),this._trigger("blur",a,{item:this.active}),this.active=null)
},_startOpening:function(a){clearTimeout(this.timer),"true"===a.attr("aria-hidden")&&(this.timer=this._delay(function(){this._close(),this._open(a)
},this.delay))
},_open:function(c){var a=k.extend({of:this.active},this.options.position);
clearTimeout(this.timer),this.element.find(".ui-menu").not(c.parents(".ui-menu")).hide().attr("aria-hidden","true"),c.show().removeAttr("aria-hidden").attr("aria-expanded","true").position(a)
},collapseAll:function(c,a){clearTimeout(this.timer),this.timer=this._delay(function(){var d=a?this.element:k(c&&c.target).closest(this.element.find(".ui-menu"));
d.length||(d=this.element),this._close(d),this.blur(c),this._removeClass(d.find(".ui-state-active"),null,"ui-state-active"),this.activeMenu=d
},this.delay)
},_close:function(a){a||(a=this.active?this.active.parent():this.element),a.find(".ui-menu").hide().attr("aria-hidden","true").attr("aria-expanded","false")
},_closeOnDocumentClick:function(a){return !k(a.target).closest(".ui-menu").length
},_isDivider:function(a){return !/[^\-\u2014\u2013\s]/.test(a.text())
},collapse:function(a){var c=this.active&&this.active.parent().closest(".ui-menu-item",this.element);
c&&c.length&&(this._close(),this.focus(a,c))
},expand:function(a){var c=this.active&&this.active.children(".ui-menu ").find(this.options.items).first();
c&&c.length&&(this._open(c.parent()),this._delay(function(){this.focus(a,c)
}))
},next:function(a){this._move("next","first",a)
},previous:function(a){this._move("prev","last",a)
},isFirstItem:function(){return this.active&&!this.active.prevAll(".ui-menu-item").length
},isLastItem:function(){return this.active&&!this.active.nextAll(".ui-menu-item").length
},_move:function(c,f,a){var d;
this.active&&(d="first"===c||"last"===c?this.active["first"===c?"prevAll":"nextAll"](".ui-menu-item").eq(-1):this.active[c+"All"](".ui-menu-item").eq(0)),d&&d.length&&this.active||(d=this.activeMenu.find(this.options.items)[f]()),this.focus(a,d)
},nextPage:function(d){var a,c,f;
return this.active?(this.isLastItem()||(this._hasScroll()?(c=this.active.offset().top,f=this.element.height(),this.active.nextAll(".ui-menu-item").each(function(){return a=k(this),0>a.offset().top-c-f
}),this.focus(d,a)):this.focus(d,this.activeMenu.find(this.options.items)[this.active?"last":"first"]())),void 0):(this.next(d),void 0)
},previousPage:function(d){var a,c,f;
return this.active?(this.isFirstItem()||(this._hasScroll()?(c=this.active.offset().top,f=this.element.height(),this.active.prevAll(".ui-menu-item").each(function(){return a=k(this),a.offset().top-c+f>0
}),this.focus(d,a)):this.focus(d,this.activeMenu.find(this.options.items).first())),void 0):(this.next(d),void 0)
},_hasScroll:function(){return this.element.outerHeight()<this.element.prop("scrollHeight")
},select:function(c){this.active=this.active||k(c.target).closest(".ui-menu-item");
var a={item:this.active};
this.active.has(".ui-menu").length||this.collapseAll(c,!0),this._trigger("select",c,a)
},_filterMenuItems:function(d){var a=d.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g,"\\$&"),c=RegExp("^"+a,"i");
return this.activeMenu.find(this.options.items).filter(".ui-menu-item").filter(function(){return c.test(k.trim(k(this).children(".ui-menu-item-wrapper").text()))
})
}}),k.widget("ui.autocomplete",{version:"1.12.1",defaultElement:"<input>",options:{appendTo:null,autoFocus:!1,delay:300,minLength:1,position:{my:"left top",at:"left bottom",collision:"none"},source:null,change:null,close:null,focus:null,open:null,response:null,search:null,select:null},requestIndex:0,pending:0,_create:function(){var g,d,f,l=this.element[0].nodeName.toLowerCase(),h="textarea"===l,c="input"===l;
this.isMultiLine=h||!c&&this._isContentEditable(this.element),this.valueMethod=this.element[h||c?"val":"text"],this.isNewMenu=!0,this._addClass("ui-autocomplete-input"),this.element.attr("autocomplete","off"),this._on(this.element,{keydown:function(e){if(this.element.prop("readOnly")){return g=!0,f=!0,d=!0,void 0
}g=!1,f=!1,d=!1;
var a=k.ui.keyCode;
switch(e.keyCode){case a.PAGE_UP:g=!0,this._move("previousPage",e);
break;
case a.PAGE_DOWN:g=!0,this._move("nextPage",e);
break;
case a.UP:g=!0,this._keyEvent("previous",e);
break;
case a.DOWN:g=!0,this._keyEvent("next",e);
break;
case a.ENTER:this.menu.active&&(g=!0,e.preventDefault(),this.menu.select(e));
break;
case a.TAB:this.menu.active&&this.menu.select(e);
break;
case a.ESCAPE:this.menu.element.is(":visible")&&(this.isMultiLine||this._value(this.term),this.close(e),e.preventDefault());
break;
default:d=!0,this._searchTimeout(e)
}},keypress:function(a){if(g){return g=!1,(!this.isMultiLine||this.menu.element.is(":visible"))&&a.preventDefault(),void 0
}if(!d){var e=k.ui.keyCode;
switch(a.keyCode){case e.PAGE_UP:this._move("previousPage",a);
break;
case e.PAGE_DOWN:this._move("nextPage",a);
break;
case e.UP:this._keyEvent("previous",a);
break;
case e.DOWN:this._keyEvent("next",a)
}}},input:function(a){return f?(f=!1,a.preventDefault(),void 0):(this._searchTimeout(a),void 0)
},focus:function(){this.selectedItem=null,this.previous=this._value()
},blur:function(a){return this.cancelBlur?(delete this.cancelBlur,void 0):(clearTimeout(this.searching),this.close(a),this._change(a),void 0)
}}),this._initSource(),this.menu=k("<ul>").appendTo(this._appendTo()).menu({role:null}).hide().menu("instance"),this._addClass(this.menu.element,"ui-autocomplete","ui-front"),this._on(this.menu.element,{mousedown:function(a){a.preventDefault(),this.cancelBlur=!0,this._delay(function(){delete this.cancelBlur,this.element[0]!==k.ui.safeActiveElement(this.document[0])&&this.element.trigger("focus")
})
},menufocus:function(o,a){var m,p;
return this.isNewMenu&&(this.isNewMenu=!1,o.originalEvent&&/^mouse/.test(o.originalEvent.type))?(this.menu.blur(),this.document.one("mousemove",function(){k(o.target).trigger(o.originalEvent)
}),void 0):(p=a.item.data("ui-autocomplete-item"),!1!==this._trigger("focus",o,{item:p})&&o.originalEvent&&/^key/.test(o.originalEvent.type)&&this._value(p.value),m=a.item.attr("aria-label")||p.value,m&&k.trim(m).length&&(this.liveRegion.children().hide(),k("<div>").text(m).appendTo(this.liveRegion)),void 0)
},menuselect:function(o,a){var m=a.item.data("ui-autocomplete-item"),p=this.previous;
this.element[0]!==k.ui.safeActiveElement(this.document[0])&&(this.element.trigger("focus"),this.previous=p,this._delay(function(){this.previous=p,this.selectedItem=m
})),!1!==this._trigger("select",o,{item:m})&&this._value(m.value),this.term=this._value(),this.close(o),this.selectedItem=m
}}),this.liveRegion=k("<div>",{role:"status","aria-live":"assertive","aria-relevant":"additions"}).appendTo(this.document[0].body),this._addClass(this.liveRegion,null,"ui-helper-hidden-accessible"),this._on(this.window,{beforeunload:function(){this.element.removeAttr("autocomplete")
}})
},_destroy:function(){clearTimeout(this.searching),this.element.removeAttr("autocomplete"),this.menu.element.remove(),this.liveRegion.remove()
},_setOption:function(a,c){this._super(a,c),"source"===a&&this._initSource(),"appendTo"===a&&this.menu.element.appendTo(this._appendTo()),"disabled"===a&&c&&this.xhr&&this.xhr.abort()
},_isEventTargetInWidget:function(c){var a=this.menu.element[0];
return c.target===this.element[0]||c.target===a||k.contains(a,c.target)
},_closeOnClickOutside:function(a){this._isEventTargetInWidget(a)||this.close()
},_appendTo:function(){var a=this.options.appendTo;
return a&&(a=a.jquery||a.nodeType?k(a):this.document.find(a).eq(0)),a&&a[0]||(a=this.element.closest(".ui-front, dialog")),a.length||(a=this.document[0].body),a
},_initSource:function(){var d,a,c=this;
k.isArray(this.options.source)?(d=this.options.source,this.source=function(e,f){f(k.ui.autocomplete.filter(d,e.term))
}):"string"==typeof this.options.source?(a=this.options.source,this.source=function(f,g){c.xhr&&c.xhr.abort(),c.xhr=k.ajax({url:a,data:f,dataType:"json",success:function(e){g(e)
},error:function(){g([])
}})
}):this.source=this.options.source
},_searchTimeout:function(a){clearTimeout(this.searching),this.searching=this._delay(function(){var f=this.term===this._value(),c=this.menu.element.is(":visible"),d=a.altKey||a.ctrlKey||a.metaKey||a.shiftKey;
(!f||f&&!c&&!d)&&(this.selectedItem=null,this.search(null,a))
},this.options.delay)
},search:function(a,c){return a=null!=a?a:this._value(),this.term=this._value(),a.length<this.options.minLength?this.close(c):this._trigger("search",c)!==!1?this._search(a):void 0
},_search:function(a){this.pending++,this._addClass("ui-autocomplete-loading"),this.cancelSearch=!1,this.source({term:a},this._response())
},_response:function(){var a=++this.requestIndex;
return k.proxy(function(c){a===this.requestIndex&&this.__response(c),this.pending--,this.pending||this._removeClass("ui-autocomplete-loading")
},this)
},__response:function(a){a&&(a=this._normalize(a)),this._trigger("response",null,{content:a}),!this.options.disabled&&a&&a.length&&!this.cancelSearch?(this._suggest(a),this._trigger("open")):this._close()
},close:function(a){this.cancelSearch=!0,this._close(a)
},_close:function(a){this._off(this.document,"mousedown"),this.menu.element.is(":visible")&&(this.menu.element.hide(),this.menu.blur(),this.isNewMenu=!0,this._trigger("close",a))
},_change:function(a){this.previous!==this._value()&&this._trigger("change",a,{item:this.selectedItem})
},_normalize:function(a){return a.length&&a[0].label&&a[0].value?a:k.map(a,function(c){return"string"==typeof c?{label:c,value:c}:k.extend({},c,{label:c.label||c.value,value:c.value||c.label})
})
},_suggest:function(c){var a=this.menu.element.empty();
this._renderMenu(a,c),this.isNewMenu=!0,this.menu.refresh(),a.show(),this._resizeMenu(),a.position(k.extend({of:this.element},this.options.position)),this.options.autoFocus&&this.menu.next(),this._on(this.document,{mousedown:"_closeOnClickOutside"})
},_resizeMenu:function(){var a=this.menu.element;
a.outerWidth(Math.max(a.width("").outerWidth()+1,this.element.outerWidth()))
},_renderMenu:function(d,a){var c=this;
k.each(a,function(f,e){c._renderItemData(d,e)
})
},_renderItemData:function(a,c){return this._renderItem(a,c).data("ui-autocomplete-item",c)
},_renderItem:function(c,a){return k("<li>").append(k("<div>").text(a.label)).appendTo(c)
},_move:function(a,c){return this.menu.element.is(":visible")?this.menu.isFirstItem()&&/^previous/.test(a)||this.menu.isLastItem()&&/^next/.test(a)?(this.isMultiLine||this._value(this.term),this.menu.blur(),void 0):(this.menu[a](c),void 0):(this.search(null,c),void 0)
},widget:function(){return this.menu.element
},_value:function(){return this.valueMethod.apply(this.element,arguments)
},_keyEvent:function(a,c){(!this.isMultiLine||this.menu.element.is(":visible"))&&(this._move(a,c),c.preventDefault())
},_isContentEditable:function(a){if(!a.length){return !1
}var c=a.prop("contentEditable");
return"inherit"===c?this._isContentEditable(a.parent()):"true"===c
}}),k.extend(k.ui.autocomplete,{escapeRegex:function(a){return a.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g,"\\$&")
},filter:function(d,a){var c=RegExp(k.ui.autocomplete.escapeRegex(a),"i");
return k.grep(d,function(e){return c.test(e.label||e.value||e)
})
}}),k.widget("ui.autocomplete",k.ui.autocomplete,{options:{messages:{noResults:"No search results.",results:function(a){return a+(a>1?" results are":" result is")+" available, use up and down arrow keys to navigate."
}}},__response:function(c){var a;
this._superApply(arguments),this.options.disabled||this.cancelSearch||(a=c&&c.length?this.options.messages.results(c.length):this.options.messages.noResults,this.liveRegion.children().hide(),k("<div>").text(a).appendTo(this.liveRegion))
}}),k.ui.autocomplete;
var E=/ui-corner-([a-z]){2,6}/g;
k.widget("ui.controlgroup",{version:"1.12.1",defaultElement:"<div>",options:{direction:"horizontal",disabled:null,onlyVisible:!0,items:{button:"input[type=button], input[type=submit], input[type=reset], button, a",controlgroupLabel:".ui-controlgroup-label",checkboxradio:"input[type='checkbox'], input[type='radio']",selectmenu:"select",spinner:".ui-spinner-input"}},_create:function(){this._enhance()
},_enhance:function(){this.element.attr("role","toolbar"),this.refresh()
},_destroy:function(){this._callChildMethod("destroy"),this.childWidgets.removeData("ui-controlgroup-data"),this.element.removeAttr("role"),this.options.items.controlgroupLabel&&this.element.find(this.options.items.controlgroupLabel).find(".ui-controlgroup-label-contents").contents().unwrap()
},_initWidgets:function(){var c=this,a=[];
k.each(this.options.items,function(e,g){var f,d={};
return g?"controlgroupLabel"===e?(f=c.element.find(g),f.each(function(){var h=k(this);
h.children(".ui-controlgroup-label-contents").length||h.contents().wrapAll("<span class='ui-controlgroup-label-contents'></span>")
}),c._addClass(f,null,"ui-widget ui-widget-content ui-state-default"),a=a.concat(f.get()),void 0):(k.fn[e]&&(d=c["_"+e+"Options"]?c["_"+e+"Options"]("middle"):{classes:{}},c.element.find(g).each(function(){var p=k(this),m=p[e]("instance"),l=k.widget.extend({},d);
if("button"!==e||!p.parent(".ui-spinner").length){m||(m=p[e]()[e]("instance")),m&&(l.classes=c._resolveClassesValues(l.classes,m)),p[e](l);
var i=p[e]("widget");
k.data(i[0],"ui-controlgroup-data",m?m:p[e]("instance")),a.push(i[0])
}})),void 0):void 0
}),this.childWidgets=k(k.unique(a)),this._addClass(this.childWidgets,"ui-controlgroup-item")
},_callChildMethod:function(a){this.childWidgets.each(function(){var c=k(this),d=c.data("ui-controlgroup-data");
d&&d[a]&&d[a]()
})
},_updateCornerClass:function(c,f){var a="ui-corner-top ui-corner-bottom ui-corner-left ui-corner-right ui-corner-all",d=this._buildSimpleOptions(f,"label").classes.label;
this._removeClass(c,null,a),this._addClass(c,null,d)
},_buildSimpleOptions:function(c,f){var a="vertical"===this.options.direction,d={classes:{}};
return d.classes[f]={middle:"",first:"ui-corner-"+(a?"top":"left"),last:"ui-corner-"+(a?"bottom":"right"),only:"ui-corner-all"}[c],d
},_spinnerOptions:function(a){var c=this._buildSimpleOptions(a,"ui-spinner");
return c.classes["ui-spinner-up"]="",c.classes["ui-spinner-down"]="",c
},_buttonOptions:function(a){return this._buildSimpleOptions(a,"ui-button")
},_checkboxradioOptions:function(a){return this._buildSimpleOptions(a,"ui-checkboxradio-label")
},_selectmenuOptions:function(a){var c="vertical"===this.options.direction;
return{width:c?"auto":!1,classes:{middle:{"ui-selectmenu-button-open":"","ui-selectmenu-button-closed":""},first:{"ui-selectmenu-button-open":"ui-corner-"+(c?"top":"tl"),"ui-selectmenu-button-closed":"ui-corner-"+(c?"top":"left")},last:{"ui-selectmenu-button-open":c?"":"ui-corner-tr","ui-selectmenu-button-closed":"ui-corner-"+(c?"bottom":"right")},only:{"ui-selectmenu-button-open":"ui-corner-top","ui-selectmenu-button-closed":"ui-corner-all"}}[a]}
},_resolveClassesValues:function(d,a){var c={};
return k.each(d,function(f){var e=a.options.classes[f]||"";
e=k.trim(e.replace(E,"")),c[f]=(e+" "+d[f]).replace(/\s+/g," ")
}),c
},_setOption:function(a,c){return"direction"===a&&this._removeClass("ui-controlgroup-"+this.options.direction),this._super(a,c),"disabled"===a?(this._callChildMethod(c?"disable":"enable"),void 0):(this.refresh(),void 0)
},refresh:function(){var c,a=this;
this._addClass("ui-controlgroup ui-controlgroup-"+this.options.direction),"horizontal"===this.options.direction&&this._addClass(null,"ui-helper-clearfix"),this._initWidgets(),c=this.childWidgets,this.options.onlyVisible&&(c=c.filter(":visible")),c.length&&(k.each(["first","last"],function(d,e){var g=c[e]().data("ui-controlgroup-data");
if(g&&a["_"+g.widgetName+"Options"]){var f=a["_"+g.widgetName+"Options"](1===c.length?"only":e);
f.classes=a._resolveClassesValues(f.classes,g),g.element[g.widgetName](f)
}else{a._updateCornerClass(c[e](),e)
}}),this._callChildMethod("refresh"))
}}),k.widget("ui.checkboxradio",[k.ui.formResetMixin,{version:"1.12.1",options:{disabled:null,label:null,icon:!0,classes:{"ui-checkboxradio-label":"ui-corner-all","ui-checkboxradio-icon":"ui-corner-all"}},_getCreateOptions:function(){var d,a,c=this,f=this._super()||{};
return this._readType(),a=this.element.labels(),this.label=k(a[a.length-1]),this.label.length||k.error("No label found for checkboxradio widget"),this.originalLabel="",this.label.contents().not(this.element[0]).each(function(){c.originalLabel+=3===this.nodeType?k(this).text():this.outerHTML
}),this.originalLabel&&(f.label=this.originalLabel),d=this.element[0].disabled,null!=d&&(f.disabled=d),f
},_create:function(){var a=this.element[0].checked;
this._bindFormResetHandler(),null==this.options.disabled&&(this.options.disabled=this.element[0].disabled),this._setOption("disabled",this.options.disabled),this._addClass("ui-checkboxradio","ui-helper-hidden-accessible"),this._addClass(this.label,"ui-checkboxradio-label","ui-button ui-widget"),"radio"===this.type&&this._addClass(this.label,"ui-checkboxradio-radio-label"),this.options.label&&this.options.label!==this.originalLabel?this._updateLabel():this.originalLabel&&(this.options.label=this.originalLabel),this._enhance(),a&&(this._addClass(this.label,"ui-checkboxradio-checked","ui-state-active"),this.icon&&this._addClass(this.icon,null,"ui-state-hover")),this._on({change:"_toggleClasses",focus:function(){this._addClass(this.label,null,"ui-state-focus ui-visual-focus")
},blur:function(){this._removeClass(this.label,null,"ui-state-focus ui-visual-focus")
}})
},_readType:function(){var a=this.element[0].nodeName.toLowerCase();
this.type=this.element[0].type,"input"===a&&/radio|checkbox/.test(this.type)||k.error("Can't create checkboxradio on element.nodeName="+a+" and element.type="+this.type)
},_enhance:function(){this._updateIcon(this.element[0].checked)
},widget:function(){return this.label
},_getRadioGroup:function(){var d,a=this.element[0].name,c="input[name='"+k.ui.escapeSelector(a)+"']";
return a?(d=this.form.length?k(this.form[0].elements).filter(c):k(c).filter(function(){return 0===k(this).form().length
}),d.not(this.element)):k([])
},_toggleClasses:function(){var a=this.element[0].checked;
this._toggleClass(this.label,"ui-checkboxradio-checked","ui-state-active",a),this.options.icon&&"checkbox"===this.type&&this._toggleClass(this.icon,null,"ui-icon-check ui-state-checked",a)._toggleClass(this.icon,null,"ui-icon-blank",!a),"radio"===this.type&&this._getRadioGroup().each(function(){var c=k(this).checkboxradio("instance");
c&&c._removeClass(c.label,"ui-checkboxradio-checked","ui-state-active")
})
},_destroy:function(){this._unbindFormResetHandler(),this.icon&&(this.icon.remove(),this.iconSpace.remove())
},_setOption:function(a,c){return"label"!==a||c?(this._super(a,c),"disabled"===a?(this._toggleClass(this.label,null,"ui-state-disabled",c),this.element[0].disabled=c,void 0):(this.refresh(),void 0)):void 0
},_updateIcon:function(c){var a="ui-icon ui-icon-background ";
this.options.icon?(this.icon||(this.icon=k("<span>"),this.iconSpace=k("<span> </span>"),this._addClass(this.iconSpace,"ui-checkboxradio-icon-space")),"checkbox"===this.type?(a+=c?"ui-icon-check ui-state-checked":"ui-icon-blank",this._removeClass(this.icon,null,c?"ui-icon-blank":"ui-icon-check")):a+="ui-icon-blank",this._addClass(this.icon,"ui-checkboxradio-icon",a),c||this._removeClass(this.icon,null,"ui-icon-check ui-state-checked"),this.icon.prependTo(this.label).after(this.iconSpace)):void 0!==this.icon&&(this.icon.remove(),this.iconSpace.remove(),delete this.icon)
},_updateLabel:function(){var a=this.label.contents().not(this.element[0]);
this.icon&&(a=a.not(this.icon[0])),this.iconSpace&&(a=a.not(this.iconSpace[0])),a.remove(),this.label.append(this.options.label)
},refresh:function(){var a=this.element[0].checked,c=this.element[0].disabled;
this._updateIcon(a),this._toggleClass(this.label,"ui-checkboxradio-checked","ui-state-active",a),null!==this.options.label&&this._updateLabel(),c!==this.options.disabled&&this._setOptions({disabled:c})
}}]),k.ui.checkboxradio,k.widget("ui.button",{version:"1.12.1",defaultElement:"<button>",options:{classes:{"ui-button":"ui-corner-all"},disabled:null,icon:null,iconPosition:"beginning",label:null,showLabel:!0},_getCreateOptions:function(){var a,c=this._super()||{};
return this.isInput=this.element.is("input"),a=this.element[0].disabled,null!=a&&(c.disabled=a),this.originalLabel=this.isInput?this.element.val():this.element.html(),this.originalLabel&&(c.label=this.originalLabel),c
},_create:function(){!this.option.showLabel&!this.options.icon&&(this.options.showLabel=!0),null==this.options.disabled&&(this.options.disabled=this.element[0].disabled||!1),this.hasTitle=!!this.element.attr("title"),this.options.label&&this.options.label!==this.originalLabel&&(this.isInput?this.element.val(this.options.label):this.element.html(this.options.label)),this._addClass("ui-button","ui-widget"),this._setOption("disabled",this.options.disabled),this._enhance(),this.element.is("a")&&this._on({keyup:function(a){a.keyCode===k.ui.keyCode.SPACE&&(a.preventDefault(),this.element[0].click?this.element[0].click():this.element.trigger("click"))
}})
},_enhance:function(){this.element.is("button")||this.element.attr("role","button"),this.options.icon&&(this._updateIcon("icon",this.options.icon),this._updateTooltip())
},_updateTooltip:function(){this.title=this.element.attr("title"),this.options.showLabel||this.title||this.element.attr("title",this.options.label)
},_updateIcon:function(d,a){var c="iconPosition"!==d,g=c?this.options.iconPosition:a,f="top"===g||"bottom"===g;
this.icon?c&&this._removeClass(this.icon,null,this.options.icon):(this.icon=k("<span>"),this._addClass(this.icon,"ui-button-icon","ui-icon"),this.options.showLabel||this._addClass("ui-button-icon-only")),c&&this._addClass(this.icon,null,a),this._attachIcon(g),f?(this._addClass(this.icon,null,"ui-widget-icon-block"),this.iconSpace&&this.iconSpace.remove()):(this.iconSpace||(this.iconSpace=k("<span> </span>"),this._addClass(this.iconSpace,"ui-button-icon-space")),this._removeClass(this.icon,null,"ui-wiget-icon-block"),this._attachIconSpace(g))
},_destroy:function(){this.element.removeAttr("role"),this.icon&&this.icon.remove(),this.iconSpace&&this.iconSpace.remove(),this.hasTitle||this.element.removeAttr("title")
},_attachIconSpace:function(a){this.icon[/^(?:end|bottom)/.test(a)?"before":"after"](this.iconSpace)
},_attachIcon:function(a){this.element[/^(?:end|bottom)/.test(a)?"append":"prepend"](this.icon)
},_setOptions:function(c){var d=void 0===c.showLabel?this.options.showLabel:c.showLabel,a=void 0===c.icon?this.options.icon:c.icon;
d||a||(c.showLabel=!0),this._super(c)
},_setOption:function(a,c){"icon"===a&&(c?this._updateIcon(a,c):this.icon&&(this.icon.remove(),this.iconSpace&&this.iconSpace.remove())),"iconPosition"===a&&this._updateIcon(a,c),"showLabel"===a&&(this._toggleClass("ui-button-icon-only",null,!c),this._updateTooltip()),"label"===a&&(this.isInput?this.element.val(c):(this.element.html(c),this.icon&&(this._attachIcon(this.options.iconPosition),this._attachIconSpace(this.options.iconPosition)))),this._super(a,c),"disabled"===a&&(this._toggleClass(null,"ui-state-disabled",c),this.element[0].disabled=c,c&&this.element.blur())
},refresh:function(){var a=this.element.is("input, button")?this.element[0].disabled:this.element.hasClass("ui-button-disabled");
a!==this.options.disabled&&this._setOptions({disabled:a}),this._updateTooltip()
}}),k.uiBackCompat!==!1&&(k.widget("ui.button",k.ui.button,{options:{text:!0,icons:{primary:null,secondary:null}},_create:function(){this.options.showLabel&&!this.options.text&&(this.options.showLabel=this.options.text),!this.options.showLabel&&this.options.text&&(this.options.text=this.options.showLabel),this.options.icon||!this.options.icons.primary&&!this.options.icons.secondary?this.options.icon&&(this.options.icons.primary=this.options.icon):this.options.icons.primary?this.options.icon=this.options.icons.primary:(this.options.icon=this.options.icons.secondary,this.options.iconPosition="end"),this._super()
},_setOption:function(a,c){return"text"===a?(this._super("showLabel",c),void 0):("showLabel"===a&&(this.options.text=c),"icon"===a&&(this.options.icons.primary=c),"icons"===a&&(c.primary?(this._super("icon",c.primary),this._super("iconPosition","beginning")):c.secondary&&(this._super("icon",c.secondary),this._super("iconPosition","end"))),this._superApply(arguments),void 0)
}}),k.fn.button=function(a){return function(){return !this.length||this.length&&"INPUT"!==this[0].tagName||this.length&&"INPUT"===this[0].tagName&&"checkbox"!==this.attr("type")&&"radio"!==this.attr("type")?a.apply(this,arguments):(k.ui.checkboxradio||k.error("Checkboxradio widget missing"),0===arguments.length?this.checkboxradio({icon:!1}):this.checkboxradio.apply(this,arguments))
}
}(k.fn.button),k.fn.buttonset=function(){return k.ui.controlgroup||k.error("Controlgroup widget missing"),"option"===arguments[0]&&"items"===arguments[1]&&arguments[2]?this.controlgroup.apply(this,[arguments[0],"items.button",arguments[2]]):"option"===arguments[0]&&"items"===arguments[1]?this.controlgroup.apply(this,[arguments[0],"items.button"]):("object"==typeof arguments[0]&&arguments[0].items&&(arguments[0].items={button:arguments[0].items}),this.controlgroup.apply(this,arguments))
}),k.ui.button,k.extend(k.ui,{datepicker:{version:"1.12.1"}});
var A;
k.extend(q.prototype,{markerClassName:"hasDatepicker",maxRows:4,_widgetDatepicker:function(){return this.dpDiv
},setDefaults:function(a){return J(this._defaults,a||{}),this
},_attachDatepicker:function(d,a){var c,g,f;
c=d.nodeName.toLowerCase(),g="div"===c||"span"===c,d.id||(this.uuid+=1,d.id="dp"+this.uuid),f=this._newInst(k(d),g),f.settings=k.extend({},a||{}),"input"===c?this._connectDatepicker(d,f):g&&this._inlineDatepicker(d,f)
},_newInst:function(d,a){var c=d[0].id.replace(/([^A-Za-z0-9_\-])/g,"\\\\$1");
return{id:c,input:d,selectedDay:0,selectedMonth:0,selectedYear:0,drawMonth:0,drawYear:0,inline:a,dpDiv:a?z(k("<div class='"+this._inlineClass+" ui-datepicker ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>")):this.dpDiv}
},_connectDatepicker:function(d,a){var c=k(d);
a.append=k([]),a.trigger=k([]),c.hasClass(this.markerClassName)||(this._attachments(c,a),c.addClass(this.markerClassName).on("keydown",this._doKeyDown).on("keypress",this._doKeyPress).on("keyup",this._doKeyUp),this._autoSize(a),k.data(d,"datepicker",a),a.settings.disabled&&this._disableDatepicker(d))
},_attachments:function(h,d){var f,m,l,c=this._get(d,"appendText"),g=this._get(d,"isRTL");
d.append&&d.append.remove(),c&&(d.append=k("<span class='"+this._appendClass+"'>"+c+"</span>"),h[g?"before":"after"](d.append)),h.off("focus",this._showDatepicker),d.trigger&&d.trigger.remove(),f=this._get(d,"showOn"),("focus"===f||"both"===f)&&h.on("focus",this._showDatepicker),("button"===f||"both"===f)&&(m=this._get(d,"buttonText"),l=this._get(d,"buttonImage"),d.trigger=k(this._get(d,"buttonImageOnly")?k("<img/>").addClass(this._triggerClass).attr({src:l,alt:m,title:m}):k("<button type='button'></button>").addClass(this._triggerClass).html(l?k("<img/>").attr({src:l,alt:m,title:m}):m)),h[g?"before":"after"](d.trigger),d.trigger.on("click",function(){return k.datepicker._datepickerShowing&&k.datepicker._lastInput===h[0]?k.datepicker._hideDatepicker():k.datepicker._datepickerShowing&&k.datepicker._lastInput!==h[0]?(k.datepicker._hideDatepicker(),k.datepicker._showDatepicker(h[0])):k.datepicker._showDatepicker(h[0]),!1
}))
},_autoSize:function(f){if(this._get(f,"autoSize")&&!f.inline){var h,d,g,m,l=new Date(2009,11,20),c=this._get(f,"dateFormat");
c.match(/[DM]/)&&(h=function(a){for(d=0,g=0,m=0;
a.length>m;
m++){a[m].length>d&&(d=a[m].length,g=m)
}return g
},l.setMonth(h(this._get(f,c.match(/MM/)?"monthNames":"monthNamesShort"))),l.setDate(h(this._get(f,c.match(/DD/)?"dayNames":"dayNamesShort"))+20-l.getDay())),f.input.attr("size",this._formatDate(f,l).length)
}},_inlineDatepicker:function(d,a){var c=k(d);
c.hasClass(this.markerClassName)||(c.addClass(this.markerClassName).append(a.dpDiv),k.data(d,"datepicker",a),this._setDate(a,this._getDefaultDate(a),!0),this._updateDatepicker(a),this._updateAlternate(a),a.settings.disabled&&this._disableDatepicker(d),a.dpDiv.css("display","block"))
},_dialogDatepicker:function(v,p,O,g,f){var a,t,m,M,N,L=this._dialogInst;
return L||(this.uuid+=1,a="dp"+this.uuid,this._dialogInput=k("<input type='text' id='"+a+"' style='position: absolute; top: -100px; width: 0px;'/>"),this._dialogInput.on("keydown",this._doKeyDown),k("body").append(this._dialogInput),L=this._dialogInst=this._newInst(this._dialogInput,!1),L.settings={},k.data(this._dialogInput[0],"datepicker",L)),J(L.settings,g||{}),p=p&&p.constructor===Date?this._formatDate(L,p):p,this._dialogInput.val(p),this._pos=f?f.length?f:[f.pageX,f.pageY]:null,this._pos||(t=document.documentElement.clientWidth,m=document.documentElement.clientHeight,M=document.documentElement.scrollLeft||document.body.scrollLeft,N=document.documentElement.scrollTop||document.body.scrollTop,this._pos=[t/2-100+M,m/2-150+N]),this._dialogInput.css("left",this._pos[0]+20+"px").css("top",this._pos[1]+"px"),L.settings.onSelect=O,this._inDialog=!0,this.dpDiv.addClass(this._dialogClass),this._showDatepicker(this._dialogInput[0]),k.blockUI&&k.blockUI(this.dpDiv),k.data(this._dialogInput[0],"datepicker",L),this
},_destroyDatepicker:function(d){var a,c=k(d),f=k.data(d,"datepicker");
c.hasClass(this.markerClassName)&&(a=d.nodeName.toLowerCase(),k.removeData(d,"datepicker"),"input"===a?(f.append.remove(),f.trigger.remove(),c.removeClass(this.markerClassName).off("focus",this._showDatepicker).off("keydown",this._doKeyDown).off("keypress",this._doKeyPress).off("keyup",this._doKeyUp)):("div"===a||"span"===a)&&c.removeClass(this.markerClassName).empty(),A===f&&(A=null))
},_enableDatepicker:function(d){var a,c,g=k(d),f=k.data(d,"datepicker");
g.hasClass(this.markerClassName)&&(a=d.nodeName.toLowerCase(),"input"===a?(d.disabled=!1,f.trigger.filter("button").each(function(){this.disabled=!1
}).end().filter("img").css({opacity:"1.0",cursor:""})):("div"===a||"span"===a)&&(c=g.children("."+this._inlineClass),c.children().removeClass("ui-state-disabled"),c.find("select.ui-datepicker-month, select.ui-datepicker-year").prop("disabled",!1)),this._disabledInputs=k.map(this._disabledInputs,function(e){return e===d?null:e
}))
},_disableDatepicker:function(d){var a,c,g=k(d),f=k.data(d,"datepicker");
g.hasClass(this.markerClassName)&&(a=d.nodeName.toLowerCase(),"input"===a?(d.disabled=!0,f.trigger.filter("button").each(function(){this.disabled=!0
}).end().filter("img").css({opacity:"0.5",cursor:"default"})):("div"===a||"span"===a)&&(c=g.children("."+this._inlineClass),c.children().addClass("ui-state-disabled"),c.find("select.ui-datepicker-month, select.ui-datepicker-year").prop("disabled",!0)),this._disabledInputs=k.map(this._disabledInputs,function(e){return e===d?null:e
}),this._disabledInputs[this._disabledInputs.length]=d)
},_isDisabledDatepicker:function(a){if(!a){return !1
}for(var c=0;
this._disabledInputs.length>c;
c++){if(this._disabledInputs[c]===a){return !0
}}return !1
},_getInst:function(c){try{return k.data(c,"datepicker")
}catch(a){throw"Missing instance data for this datepicker"
}},_optionDatepicker:function(m,c,f){var t,p,g,d,a=this._getInst(m);
return 2===arguments.length&&"string"==typeof c?"defaults"===c?k.extend({},k.datepicker._defaults):a?"all"===c?k.extend({},a.settings):this._get(a,c):null:(t=c||{},"string"==typeof c&&(t={},t[c]=f),a&&(this._curInst===a&&this._hideDatepicker(),p=this._getDateDatepicker(m,!0),g=this._getMinMaxDate(a,"min"),d=this._getMinMaxDate(a,"max"),J(a.settings,t),null!==g&&void 0!==t.dateFormat&&void 0===t.minDate&&(a.settings.minDate=this._formatDate(a,g)),null!==d&&void 0!==t.dateFormat&&void 0===t.maxDate&&(a.settings.maxDate=this._formatDate(a,d)),"disabled" in t&&(t.disabled?this._disableDatepicker(m):this._enableDatepicker(m)),this._attachments(k(m),a),this._autoSize(a),this._setDate(a,p),this._updateAlternate(a),this._updateDatepicker(a)),void 0)
},_changeDatepicker:function(c,d,a){this._optionDatepicker(c,d,a)
},_refreshDatepicker:function(a){var c=this._getInst(a);
c&&this._updateDatepicker(c)
},_setDateDatepicker:function(c,d){var a=this._getInst(c);
a&&(this._setDate(a,d),this._updateDatepicker(a),this._updateAlternate(a))
},_getDateDatepicker:function(c,d){var a=this._getInst(c);
return a&&!a.inline&&this._setDateFromField(a,d),a?this._getDate(a):null
},_doKeyDown:function(h){var d,f,m,l=k.datepicker._getInst(h.target),c=!0,g=l.dpDiv.is(".ui-datepicker-rtl");
if(l._keyEvent=!0,k.datepicker._datepickerShowing){switch(h.keyCode){case 9:k.datepicker._hideDatepicker(),c=!1;
break;
case 13:return m=k("td."+k.datepicker._dayOverClass+":not(."+k.datepicker._currentClass+")",l.dpDiv),m[0]&&k.datepicker._selectDay(h.target,l.selectedMonth,l.selectedYear,m[0]),d=k.datepicker._get(l,"onSelect"),d?(f=k.datepicker._formatDate(l),d.apply(l.input?l.input[0]:null,[f,l])):k.datepicker._hideDatepicker(),!1;
case 27:k.datepicker._hideDatepicker();
break;
case 33:k.datepicker._adjustDate(h.target,h.ctrlKey?-k.datepicker._get(l,"stepBigMonths"):-k.datepicker._get(l,"stepMonths"),"M");
break;
case 34:k.datepicker._adjustDate(h.target,h.ctrlKey?+k.datepicker._get(l,"stepBigMonths"):+k.datepicker._get(l,"stepMonths"),"M");
break;
case 35:(h.ctrlKey||h.metaKey)&&k.datepicker._clearDate(h.target),c=h.ctrlKey||h.metaKey;
break;
case 36:(h.ctrlKey||h.metaKey)&&k.datepicker._gotoToday(h.target),c=h.ctrlKey||h.metaKey;
break;
case 37:(h.ctrlKey||h.metaKey)&&k.datepicker._adjustDate(h.target,g?1:-1,"D"),c=h.ctrlKey||h.metaKey,h.originalEvent.altKey&&k.datepicker._adjustDate(h.target,h.ctrlKey?-k.datepicker._get(l,"stepBigMonths"):-k.datepicker._get(l,"stepMonths"),"M");
break;
case 38:(h.ctrlKey||h.metaKey)&&k.datepicker._adjustDate(h.target,-7,"D"),c=h.ctrlKey||h.metaKey;
break;
case 39:(h.ctrlKey||h.metaKey)&&k.datepicker._adjustDate(h.target,g?-1:1,"D"),c=h.ctrlKey||h.metaKey,h.originalEvent.altKey&&k.datepicker._adjustDate(h.target,h.ctrlKey?+k.datepicker._get(l,"stepBigMonths"):+k.datepicker._get(l,"stepMonths"),"M");
break;
case 40:(h.ctrlKey||h.metaKey)&&k.datepicker._adjustDate(h.target,7,"D"),c=h.ctrlKey||h.metaKey;
break;
default:c=!1
}}else{36===h.keyCode&&h.ctrlKey?k.datepicker._showDatepicker(this):c=!1
}c&&(h.preventDefault(),h.stopPropagation())
},_doKeyPress:function(d){var a,c,f=k.datepicker._getInst(d.target);
return k.datepicker._get(f,"constrainInput")?(a=k.datepicker._possibleChars(k.datepicker._get(f,"dateFormat")),c=String.fromCharCode(null==d.charCode?d.keyCode:d.charCode),d.ctrlKey||d.metaKey||" ">c||!a||a.indexOf(c)>-1):void 0
},_doKeyUp:function(d){var a,c=k.datepicker._getInst(d.target);
if(c.input.val()!==c.lastVal){try{a=k.datepicker.parseDate(k.datepicker._get(c,"dateFormat"),c.input?c.input.val():null,k.datepicker._getFormatConfig(c)),a&&(k.datepicker._setDateFromField(c),k.datepicker._updateAlternate(c),k.datepicker._updateDatepicker(c))
}catch(f){}}return !0
},_showDatepicker:function(i){if(i=i.target||i,"input"!==i.nodeName.toLowerCase()&&(i=k("input",i.parentNode)[0]),!k.datepicker._isDisabledDatepicker(i)&&k.datepicker._lastInput!==i){var f,t,m,g,d,a,p;
f=k.datepicker._getInst(i),k.datepicker._curInst&&k.datepicker._curInst!==f&&(k.datepicker._curInst.dpDiv.stop(!0,!0),f&&k.datepicker._datepickerShowing&&k.datepicker._hideDatepicker(k.datepicker._curInst.input[0])),t=k.datepicker._get(f,"beforeShow"),m=t?t.apply(i,[i,f]):{},m!==!1&&(J(f.settings,m),f.lastVal=null,k.datepicker._lastInput=i,k.datepicker._setDateFromField(f),k.datepicker._inDialog&&(i.value=""),k.datepicker._pos||(k.datepicker._pos=k.datepicker._findPos(i),k.datepicker._pos[1]+=i.offsetHeight),g=!1,k(i).parents().each(function(){return g|="fixed"===k(this).css("position"),!g
}),d={left:k.datepicker._pos[0],top:k.datepicker._pos[1]},k.datepicker._pos=null,f.dpDiv.empty(),f.dpDiv.css({position:"absolute",display:"block",top:"-1000px"}),k.datepicker._updateDatepicker(f),d=k.datepicker._checkOffset(f,d,g),f.dpDiv.css({position:k.datepicker._inDialog&&k.blockUI?"static":g?"fixed":"absolute",display:"none",left:d.left+"px",top:d.top+"px"}),f.inline||(a=k.datepicker._get(f,"showAnim"),p=k.datepicker._get(f,"duration"),f.dpDiv.css("z-index",C(k(i))+1),k.datepicker._datepickerShowing=!0,k.effects&&k.effects.effect[a]?f.dpDiv.show(a,k.datepicker._get(f,"showOptions"),p):f.dpDiv[a||"show"](a?p:null),k.datepicker._shouldFocusInput(f)&&f.input.trigger("focus"),k.datepicker._curInst=f))
}},_updateDatepicker:function(h){this.maxRows=4,A=h,h.dpDiv.empty().append(this._generateHTML(h)),this._attachHandlers(h);
var d,f=this._getNumberOfMonths(h),l=f[1],c=17,g=h.dpDiv.find("."+this._dayOverClass+" a");
g.length>0&&y.apply(g.get(0)),h.dpDiv.removeClass("ui-datepicker-multi-2 ui-datepicker-multi-3 ui-datepicker-multi-4").width(""),l>1&&h.dpDiv.addClass("ui-datepicker-multi-"+l).css("width",c*l+"em"),h.dpDiv[(1!==f[0]||1!==f[1]?"add":"remove")+"Class"]("ui-datepicker-multi"),h.dpDiv[(this._get(h,"isRTL")?"add":"remove")+"Class"]("ui-datepicker-rtl"),h===k.datepicker._curInst&&k.datepicker._datepickerShowing&&k.datepicker._shouldFocusInput(h)&&h.input.trigger("focus"),h.yearshtml&&(d=h.yearshtml,setTimeout(function(){d===h.yearshtml&&h.yearshtml&&h.dpDiv.find("select.ui-datepicker-year:first").replaceWith(h.yearshtml),d=h.yearshtml=null
},0))
},_shouldFocusInput:function(a){return a.input&&a.input.is(":visible")&&!a.input.is(":disabled")&&!a.input.is(":focus")
},_checkOffset:function(t,m,v){var f=t.dpDiv.outerWidth(),d=t.dpDiv.outerHeight(),u=t.input?t.input.outerWidth():0,c=t.input?t.input.outerHeight():0,p=document.documentElement.clientWidth+(v?0:k(document).scrollLeft()),g=document.documentElement.clientHeight+(v?0:k(document).scrollTop());
return m.left-=this._get(t,"isRTL")?f-u:0,m.left-=v&&m.left===t.input.offset().left?k(document).scrollLeft():0,m.top-=v&&m.top===t.input.offset().top+c?k(document).scrollTop():0,m.left-=Math.min(m.left,m.left+f>p&&p>f?Math.abs(m.left+f-p):0),m.top-=Math.min(m.top,m.top+d>g&&g>d?Math.abs(d+c):0),m
},_findPos:function(d){for(var a,c=this._getInst(d),f=this._get(c,"isRTL");
d&&("hidden"===d.type||1!==d.nodeType||k.expr.filters.hidden(d));
){d=d[f?"previousSibling":"nextSibling"]
}return a=k(d).offset(),[a.left,a.top]
},_hideDatepicker:function(g){var d,f,l,h,c=this._curInst;
!c||g&&c!==k.data(g,"datepicker")||this._datepickerShowing&&(d=this._get(c,"showAnim"),f=this._get(c,"duration"),l=function(){k.datepicker._tidyDialog(c)
},k.effects&&(k.effects.effect[d]||k.effects[d])?c.dpDiv.hide(d,k.datepicker._get(c,"showOptions"),f,l):c.dpDiv["slideDown"===d?"slideUp":"fadeIn"===d?"fadeOut":"hide"](d?f:null,l),d||l(),this._datepickerShowing=!1,h=this._get(c,"onClose"),h&&h.apply(c.input?c.input[0]:null,[c.input?c.input.val():"",c]),this._lastInput=null,this._inDialog&&(this._dialogInput.css({position:"absolute",left:"0",top:"-100px"}),k.blockUI&&(k.unblockUI(),k("body").append(this.dpDiv))),this._inDialog=!1)
},_tidyDialog:function(a){a.dpDiv.removeClass(this._dialogClass).off(".ui-datepicker-calendar")
},_checkExternalClick:function(d){if(k.datepicker._curInst){var a=k(d.target),c=k.datepicker._getInst(a[0]);
(a[0].id!==k.datepicker._mainDivId&&0===a.parents("#"+k.datepicker._mainDivId).length&&!a.hasClass(k.datepicker.markerClassName)&&!a.closest("."+k.datepicker._triggerClass).length&&k.datepicker._datepickerShowing&&(!k.datepicker._inDialog||!k.blockUI)||a.hasClass(k.datepicker.markerClassName)&&k.datepicker._curInst!==c)&&k.datepicker._hideDatepicker()
}},_adjustDate:function(d,a,c){var g=k(d),f=this._getInst(g[0]);
this._isDisabledDatepicker(g[0])||(this._adjustInstDate(f,a+("M"===c?this._get(f,"showCurrentAtPos"):0),c),this._updateDatepicker(f))
},_gotoToday:function(d){var a,c=k(d),f=this._getInst(c[0]);
this._get(f,"gotoCurrent")&&f.currentDay?(f.selectedDay=f.currentDay,f.drawMonth=f.selectedMonth=f.currentMonth,f.drawYear=f.selectedYear=f.currentYear):(a=new Date,f.selectedDay=a.getDate(),f.drawMonth=f.selectedMonth=a.getMonth(),f.drawYear=f.selectedYear=a.getFullYear()),this._notifyChange(f),this._adjustDate(c)
},_selectMonthYear:function(d,a,c){var g=k(d),f=this._getInst(g[0]);
f["selected"+("M"===c?"Month":"Year")]=f["draw"+("M"===c?"Month":"Year")]=parseInt(a.options[a.selectedIndex].value,10),this._notifyChange(f),this._adjustDate(g)
},_selectDay:function(g,d,f,l){var h,c=k(g);
k(l).hasClass(this._unselectableClass)||this._isDisabledDatepicker(c[0])||(h=this._getInst(c[0]),h.selectedDay=h.currentDay=k("a",l).html(),h.selectedMonth=h.currentMonth=d,h.selectedYear=h.currentYear=f,this._selectDate(g,this._formatDate(h,h.currentDay,h.currentMonth,h.currentYear)))
},_clearDate:function(c){var a=k(c);
this._selectDate(a,"")
},_selectDate:function(d,a){var c,g=k(d),f=this._getInst(g[0]);
a=null!=a?a:this._formatDate(f),f.input&&f.input.val(a),this._updateAlternate(f),c=this._get(f,"onSelect"),c?c.apply(f.input?f.input[0]:null,[a,f]):f.input&&f.input.trigger("change"),f.inline?this._updateDatepicker(f):(this._hideDatepicker(),this._lastInput=f.input[0],"object"!=typeof f.input[0]&&f.input.trigger("focus"),this._lastInput=null)
},_updateAlternate:function(d){var a,c,g,f=this._get(d,"altField");
f&&(a=this._get(d,"altFormat")||this._get(d,"dateFormat"),c=this._getDate(d),g=this.formatDate(a,c,this._getFormatConfig(d)),k(f).val(g))
},noWeekends:function(a){var c=a.getDay();
return[c>0&&6>c,""]
},iso8601Week:function(c){var d,a=new Date(c.getTime());
return a.setDate(a.getDate()+4-(a.getDay()||7)),d=a.getTime(),a.setMonth(0),a.setDate(1),Math.floor(Math.round((d-a)/86400000)/7)+1
},parseDate:function(ab,X,P){if(null==ab||null==X){throw"Invalid arguments"
}if(X="object"==typeof X?""+X:X+"",""===X){return null
}var T,S,af,Q,Y=0,V=(P?P.shortYearCutoff:null)||this._defaults.shortYearCutoff,ad="string"!=typeof V?V:(new Date).getFullYear()%100+parseInt(V,10),O=(P?P.dayNamesShort:null)||this._defaults.dayNamesShort,ac=(P?P.dayNames:null)||this._defaults.dayNames,R=(P?P.monthNamesShort:null)||this._defaults.monthNamesShort,aa=(P?P.monthNames:null)||this._defaults.monthNames,Z=-1,U=-1,ag=-1,N=-1,ae=!1,t=function(c){var a=ab.length>T+1&&ab.charAt(T+1)===c;
return a&&T++,a
},M=function(d){var g=t(d),f="@"===d?14:"!"===d?20:"y"===d&&g?4:"o"===d?3:2,i="y"===d?f:1,h=RegExp("^\\d{"+i+","+f+"}"),c=X.substring(Y).match(h);
if(!c){throw"Missing number at position "+Y
}return Y+=c[0].length,parseInt(c[0],10)
},W=function(f,d,h){var g=-1,c=k.map(t(f)?h:d,function(a,i){return[[i,a]]
}).sort(function(a,i){return -(a[1].length-i[1].length)
});
if(k.each(c,function(a,l){var i=l[1];
return X.substr(Y,i.length).toLowerCase()===i.toLowerCase()?(g=l[0],Y+=i.length,!1):void 0
}),-1!==g){return g+1
}throw"Unknown name at position "+Y
},L=function(){if(X.charAt(Y)!==ab.charAt(T)){throw"Unexpected literal at position "+Y
}Y++
};
for(T=0;
ab.length>T;
T++){if(ae){"'"!==ab.charAt(T)||t("'")?L():ae=!1
}else{switch(ab.charAt(T)){case"d":ag=M("d");
break;
case"D":W("D",O,ac);
break;
case"o":N=M("o");
break;
case"m":U=M("m");
break;
case"M":U=W("M",R,aa);
break;
case"y":Z=M("y");
break;
case"@":Q=new Date(M("@")),Z=Q.getFullYear(),U=Q.getMonth()+1,ag=Q.getDate();
break;
case"!":Q=new Date((M("!")-this._ticksTo1970)/10000),Z=Q.getFullYear(),U=Q.getMonth()+1,ag=Q.getDate();
break;
case"'":t("'")?L():ae=!0;
break;
default:L()
}}}if(X.length>Y&&(af=X.substr(Y),!/^\s+/.test(af))){throw"Extra/unparsed characters found in date: "+af
}if(-1===Z?Z=(new Date).getFullYear():100>Z&&(Z+=(new Date).getFullYear()-(new Date).getFullYear()%100+(ad>=Z?0:-100)),N>-1){for(U=1,ag=N;
;
){if(S=this._getDaysInMonth(Z,U-1),S>=ag){break
}U++,ag-=S
}}if(Q=this._daylightSavingAdjust(new Date(Z,U-1,ag)),Q.getFullYear()!==Z||Q.getMonth()+1!==U||Q.getDate()!==ag){throw"Invalid date"
}return Q
},ATOM:"yy-mm-dd",COOKIE:"D, dd M yy",ISO_8601:"yy-mm-dd",RFC_822:"D, d M y",RFC_850:"DD, dd-M-y",RFC_1036:"D, d M y",RFC_1123:"D, d M yy",RFC_2822:"D, d M yy",RSS:"D, d M y",TICKS:"!",TIMESTAMP:"@",W3C:"yy-mm-dd",_ticksTo1970:10000000*60*60*24*(718685+Math.floor(492.5)-Math.floor(19.7)+Math.floor(4.925)),formatDate:function(R,M,v){if(!M){return""
}var S,m=(v?v.dayNamesShort:null)||this._defaults.dayNamesShort,g=(v?v.dayNames:null)||this._defaults.dayNames,P=(v?v.monthNamesShort:null)||this._defaults.monthNamesShort,f=(v?v.monthNames:null)||this._defaults.monthNames,L=function(c){var a=R.length>S+1&&R.charAt(S+1)===c;
return a&&S++,a
},p=function(c,h,a){var d=""+h;
if(L(c)){for(;
a>d.length;
){d="0"+d
}}return d
},O=function(c,h,a,d){return L(c)?d[h]:a[h]
},Q="",N=!1;
if(M){for(S=0;
R.length>S;
S++){if(N){"'"!==R.charAt(S)||L("'")?Q+=R.charAt(S):N=!1
}else{switch(R.charAt(S)){case"d":Q+=p("d",M.getDate(),2);
break;
case"D":Q+=O("D",M.getDay(),m,g);
break;
case"o":Q+=p("o",Math.round((new Date(M.getFullYear(),M.getMonth(),M.getDate()).getTime()-new Date(M.getFullYear(),0,0).getTime())/86400000),3);
break;
case"m":Q+=p("m",M.getMonth()+1,2);
break;
case"M":Q+=O("M",M.getMonth(),P,f);
break;
case"y":Q+=L("y")?M.getFullYear():(10>M.getFullYear()%100?"0":"")+M.getFullYear()%100;
break;
case"@":Q+=M.getTime();
break;
case"!":Q+=10000*M.getTime()+this._ticksTo1970;
break;
case"'":L("'")?Q+="'":N=!0;
break;
default:Q+=R.charAt(S)
}}}}return Q
},_possibleChars:function(c){var f,a="",d=!1,g=function(e){var h=c.length>f+1&&c.charAt(f+1)===e;
return h&&f++,h
};
for(f=0;
c.length>f;
f++){if(d){"'"!==c.charAt(f)||g("'")?a+=c.charAt(f):d=!1
}else{switch(c.charAt(f)){case"d":case"m":case"y":case"@":a+="0123456789";
break;
case"D":case"M":return null;
case"'":g("'")?a+="'":d=!0;
break;
default:a+=c.charAt(f)
}}}return a
},_get:function(a,c){return void 0!==a.settings[c]?a.settings[c]:this._defaults[c]
},_setDateFromField:function(f,l){if(f.input.val()!==f.lastVal){var d=this._get(f,"dateFormat"),g=f.lastVal=f.input?f.input.val():null,p=this._getDefaultDate(f),m=p,c=this._getFormatConfig(f);
try{m=this.parseDate(d,g,c)||p
}catch(h){g=l?"":g
}f.selectedDay=m.getDate(),f.drawMonth=f.selectedMonth=m.getMonth(),f.drawYear=f.selectedYear=m.getFullYear(),f.currentDay=g?m.getDate():0,f.currentMonth=g?m.getMonth():0,f.currentYear=g?m.getFullYear():0,this._adjustInstDate(f)
}},_getDefaultDate:function(a){return this._restrictMinMax(a,this._determineDate(a,this._get(a,"defaultDate"),new Date))
},_determineDate:function(g,d,f){var l=function(a){var i=new Date;
return i.setDate(i.getDate()+a),i
},h=function(p){try{return k.datepicker.parseDate(k.datepicker._get(g,"dateFormat"),p,k.datepicker._getFormatConfig(g))
}catch(u){}for(var M=(p.toLowerCase().match(/^c/)?k.datepicker._getDate(g):null)||new Date,L=M.getFullYear(),m=M.getMonth(),v=M.getDate(),t=/([+\-]?[0-9]+)\s*(d|D|w|W|m|M|y|Y)?/g,e=t.exec(p);
e;
){switch(e[2]||"d"){case"d":case"D":v+=parseInt(e[1],10);
break;
case"w":case"W":v+=7*parseInt(e[1],10);
break;
case"m":case"M":m+=parseInt(e[1],10),v=Math.min(v,k.datepicker._getDaysInMonth(L,m));
break;
case"y":case"Y":L+=parseInt(e[1],10),v=Math.min(v,k.datepicker._getDaysInMonth(L,m))
}e=t.exec(p)
}return new Date(L,m,v)
},c=null==d||""===d?f:"string"==typeof d?h(d):"number"==typeof d?isNaN(d)?f:l(d):new Date(d.getTime());
return c=c&&"Invalid Date"==""+c?f:c,c&&(c.setHours(0),c.setMinutes(0),c.setSeconds(0),c.setMilliseconds(0)),this._daylightSavingAdjust(c)
},_daylightSavingAdjust:function(a){return a?(a.setHours(a.getHours()>12?a.getHours()+2:0),a):null
},_setDate:function(f,h,d){var g=!h,m=f.selectedMonth,l=f.selectedYear,c=this._restrictMinMax(f,this._determineDate(f,h,new Date));
f.selectedDay=f.currentDay=c.getDate(),f.drawMonth=f.selectedMonth=f.currentMonth=c.getMonth(),f.drawYear=f.selectedYear=f.currentYear=c.getFullYear(),m===f.selectedMonth&&l===f.selectedYear||d||this._notifyChange(f),this._adjustInstDate(f),f.input&&f.input.val(g?"":this._formatDate(f))
},_getDate:function(a){var c=!a.currentYear||a.input&&""===a.input.val()?null:this._daylightSavingAdjust(new Date(a.currentYear,a.currentMonth,a.currentDay));
return c
},_attachHandlers:function(d){var a=this._get(d,"stepMonths"),c="#"+d.id.replace(/\\\\/g,"\\");
d.dpDiv.find("[data-handler]").map(function(){var f={prev:function(){k.datepicker._adjustDate(c,-a,"M")
},next:function(){k.datepicker._adjustDate(c,+a,"M")
},hide:function(){k.datepicker._hideDatepicker()
},today:function(){k.datepicker._gotoToday(c)
},selectDay:function(){return k.datepicker._selectDay(c,+this.getAttribute("data-month"),+this.getAttribute("data-year"),this),!1
},selectMonth:function(){return k.datepicker._selectMonthYear(c,this,"M"),!1
},selectYear:function(){return k.datepicker._selectMonthYear(c,this,"Y"),!1
}};
k(this).on(this.getAttribute("data-event"),f[this.getAttribute("data-handler")])
})
},_generateHTML:function(aJ){var aY,aU,aK,aP,aO,a2,aL,aV,aR,a0,aI,aZ,aN,aX,aW,aQ,a3,aH,a1,aE,aG,aS,aF,az,ay,at,ah,al,ao,ai,au,aD,am,aB,an,ae,ax,aw,ap,aj=new Date,aA=this._daylightSavingAdjust(new Date(aj.getFullYear(),aj.getMonth(),aj.getDate())),ab=this._get(aJ,"isRTL"),aT=this._get(aJ,"showButtonPanel"),aM=this._get(aJ,"hideIfNoPrevNext"),aq=this._get(aJ,"navigationAsDateFormat"),ag=this._getNumberOfMonths(aJ),af=this._get(aJ,"showCurrentAtPos"),aC=this._get(aJ,"stepMonths"),ac=1!==ag[0]||1!==ag[1],av=this._daylightSavingAdjust(aJ.currentDay?new Date(aJ.currentYear,aJ.currentMonth,aJ.currentDay):new Date(9999,9,9)),ak=this._getMinMaxDate(aJ,"min"),ar=this._getMinMaxDate(aJ,"max"),aa=aJ.drawMonth-af,ad=aJ.drawYear;
if(0>aa&&(aa+=12,ad--),ar){for(aY=this._daylightSavingAdjust(new Date(ar.getFullYear(),ar.getMonth()-ag[0]*ag[1]+1,ar.getDate())),aY=ak&&ak>aY?ak:aY;
this._daylightSavingAdjust(new Date(ad,aa,1))>aY;
){aa--,0>aa&&(aa=11,ad--)
}}for(aJ.drawMonth=aa,aJ.drawYear=ad,aU=this._get(aJ,"prevText"),aU=aq?this.formatDate(aU,this._daylightSavingAdjust(new Date(ad,aa-aC,1)),this._getFormatConfig(aJ)):aU,aK=this._canAdjustMonth(aJ,-1,ad,aa)?"<a class='ui-datepicker-prev ui-corner-all' data-handler='prev' data-event='click' title='"+aU+"'><span class='ui-icon ui-icon-circle-triangle-"+(ab?"e":"w")+"'>"+aU+"</span></a>":aM?"":"<a class='ui-datepicker-prev ui-corner-all ui-state-disabled' title='"+aU+"'><span class='ui-icon ui-icon-circle-triangle-"+(ab?"e":"w")+"'>"+aU+"</span></a>",aP=this._get(aJ,"nextText"),aP=aq?this.formatDate(aP,this._daylightSavingAdjust(new Date(ad,aa+aC,1)),this._getFormatConfig(aJ)):aP,aO=this._canAdjustMonth(aJ,1,ad,aa)?"<a class='ui-datepicker-next ui-corner-all' data-handler='next' data-event='click' title='"+aP+"'><span class='ui-icon ui-icon-circle-triangle-"+(ab?"w":"e")+"'>"+aP+"</span></a>":aM?"":"<a class='ui-datepicker-next ui-corner-all ui-state-disabled' title='"+aP+"'><span class='ui-icon ui-icon-circle-triangle-"+(ab?"w":"e")+"'>"+aP+"</span></a>",a2=this._get(aJ,"currentText"),aL=this._get(aJ,"gotoCurrent")&&aJ.currentDay?av:aA,a2=aq?this.formatDate(a2,aL,this._getFormatConfig(aJ)):a2,aV=aJ.inline?"":"<button type='button' class='ui-datepicker-close ui-state-default ui-priority-primary ui-corner-all' data-handler='hide' data-event='click'>"+this._get(aJ,"closeText")+"</button>",aR=aT?"<div class='ui-datepicker-buttonpane ui-widget-content'>"+(ab?aV:"")+(this._isInRange(aJ,aL)?"<button type='button' class='ui-datepicker-current ui-state-default ui-priority-secondary ui-corner-all' data-handler='today' data-event='click'>"+a2+"</button>":"")+(ab?"":aV)+"</div>":"",a0=parseInt(this._get(aJ,"firstDay"),10),a0=isNaN(a0)?0:a0,aI=this._get(aJ,"showWeek"),aZ=this._get(aJ,"dayNames"),aN=this._get(aJ,"dayNamesMin"),aX=this._get(aJ,"monthNames"),aW=this._get(aJ,"monthNamesShort"),aQ=this._get(aJ,"beforeShowDay"),a3=this._get(aJ,"showOtherMonths"),aH=this._get(aJ,"selectOtherMonths"),a1=this._getDefaultDate(aJ),aE="",aS=0;
ag[0]>aS;
aS++){for(aF="",this.maxRows=4,az=0;
ag[1]>az;
az++){if(ay=this._daylightSavingAdjust(new Date(ad,aa,aJ.selectedDay)),at=" ui-corner-all",ah="",ac){if(ah+="<div class='ui-datepicker-group",ag[1]>1){switch(az){case 0:ah+=" ui-datepicker-group-first",at=" ui-corner-"+(ab?"right":"left");
break;
case ag[1]-1:ah+=" ui-datepicker-group-last",at=" ui-corner-"+(ab?"left":"right");
break;
default:ah+=" ui-datepicker-group-middle",at=""
}}ah+="'>"
}for(ah+="<div class='ui-datepicker-header ui-widget-header ui-helper-clearfix"+at+"'>"+(/all|left/.test(at)&&0===aS?ab?aO:aK:"")+(/all|right/.test(at)&&0===aS?ab?aK:aO:"")+this._generateMonthYearHeader(aJ,aa,ad,ak,ar,aS>0||az>0,aX,aW)+"</div><table class='ui-datepicker-calendar'><thead><tr>",al=aI?"<th class='ui-datepicker-week-col'>"+this._get(aJ,"weekHeader")+"</th>":"",aG=0;
7>aG;
aG++){ao=(aG+a0)%7,al+="<th scope='col'"+((aG+a0+6)%7>=5?" class='ui-datepicker-week-end'":"")+"><span title='"+aZ[ao]+"'>"+aN[ao]+"</span></th>"
}for(ah+=al+"</tr></thead><tbody>",ai=this._getDaysInMonth(ad,aa),ad===aJ.selectedYear&&aa===aJ.selectedMonth&&(aJ.selectedDay=Math.min(aJ.selectedDay,ai)),au=(this._getFirstDayOfMonth(ad,aa)-a0+7)%7,aD=Math.ceil((au+ai)/7),am=ac?this.maxRows>aD?this.maxRows:aD:aD,this.maxRows=am,aB=this._daylightSavingAdjust(new Date(ad,aa,1-au)),an=0;
am>an;
an++){for(ah+="<tr>",ae=aI?"<td class='ui-datepicker-week-col'>"+this._get(aJ,"calculateWeek")(aB)+"</td>":"",aG=0;
7>aG;
aG++){ax=aQ?aQ.apply(aJ.input?aJ.input[0]:null,[aB]):[!0,""],aw=aB.getMonth()!==aa,ap=aw&&!aH||!ax[0]||ak&&ak>aB||ar&&aB>ar,ae+="<td class='"+((aG+a0+6)%7>=5?" ui-datepicker-week-end":"")+(aw?" ui-datepicker-other-month":"")+(aB.getTime()===ay.getTime()&&aa===aJ.selectedMonth&&aJ._keyEvent||a1.getTime()===aB.getTime()&&a1.getTime()===ay.getTime()?" "+this._dayOverClass:"")+(ap?" "+this._unselectableClass+" ui-state-disabled":"")+(aw&&!a3?"":" "+ax[1]+(aB.getTime()===av.getTime()?" "+this._currentClass:"")+(aB.getTime()===aA.getTime()?" ui-datepicker-today":""))+"'"+(aw&&!a3||!ax[2]?"":" title='"+ax[2].replace(/'/g,"&#39;")+"'")+(ap?"":" data-handler='selectDay' data-event='click' data-month='"+aB.getMonth()+"' data-year='"+aB.getFullYear()+"'")+">"+(aw&&!a3?"&#xa0;":ap?"<span class='ui-state-default'>"+aB.getDate()+"</span>":"<a class='ui-state-default"+(aB.getTime()===aA.getTime()?" ui-state-highlight":"")+(aB.getTime()===av.getTime()?" ui-state-active":"")+(aw?" ui-priority-secondary":"")+"' href='#'>"+aB.getDate()+"</a>")+"</td>",aB.setDate(aB.getDate()+1),aB=this._daylightSavingAdjust(aB)
}ah+=ae+"</tr>"
}aa++,aa>11&&(aa=0,ad++),ah+="</tbody></table>"+(ac?"</div>"+(ag[0]>0&&az===ag[1]-1?"<div class='ui-datepicker-row-break'></div>":""):""),aF+=ah
}aE+=aF
}return aE+=aR,aJ._keyEvent=!1,aE
},_generateMonthYearHeader:function(O,aa,W,P,T,S,ae,Q){var X,V,ac,N,ab,R,Z,Y,U=this._get(O,"changeMonth"),af=this._get(O,"changeYear"),M=this._get(O,"showMonthAfterYear"),ad="<div class='ui-datepicker-title'>",L="";
if(S||!U){L+="<span class='ui-datepicker-month'>"+ae[aa]+"</span>"
}else{for(X=P&&P.getFullYear()===W,V=T&&T.getFullYear()===W,L+="<select class='ui-datepicker-month' data-handler='selectMonth' data-event='change'>",ac=0;
12>ac;
ac++){(!X||ac>=P.getMonth())&&(!V||T.getMonth()>=ac)&&(L+="<option value='"+ac+"'"+(ac===aa?" selected='selected'":"")+">"+Q[ac]+"</option>")
}L+="</select>"
}if(M||(ad+=L+(!S&&U&&af?"":"&#xa0;")),!O.yearshtml){if(O.yearshtml="",S||!af){ad+="<span class='ui-datepicker-year'>"+W+"</span>"
}else{for(N=this._get(O,"yearRange").split(":"),ab=(new Date).getFullYear(),R=function(a){var c=a.match(/c[+\-].*/)?W+parseInt(a.substring(1),10):a.match(/[+\-].*/)?ab+parseInt(a,10):parseInt(a,10);
return isNaN(c)?ab:c
},Z=R(N[0]),Y=Math.max(Z,R(N[1]||"")),Z=P?Math.max(Z,P.getFullYear()):Z,Y=T?Math.min(Y,T.getFullYear()):Y,O.yearshtml+="<select class='ui-datepicker-year' data-handler='selectYear' data-event='change'>";
Y>=Z;
Z++){O.yearshtml+="<option value='"+Z+"'"+(Z===W?" selected='selected'":"")+">"+Z+"</option>"
}O.yearshtml+="</select>",ad+=O.yearshtml,O.yearshtml=null
}}return ad+=this._get(O,"yearSuffix"),M&&(ad+=(!S&&U&&af?"":"&#xa0;")+L),ad+="</div>"
},_adjustInstDate:function(f,h,d){var g=f.selectedYear+("Y"===d?h:0),m=f.selectedMonth+("M"===d?h:0),l=Math.min(f.selectedDay,this._getDaysInMonth(g,m))+("D"===d?h:0),c=this._restrictMinMax(f,this._daylightSavingAdjust(new Date(g,m,l)));
f.selectedDay=c.getDate(),f.drawMonth=f.selectedMonth=c.getMonth(),f.drawYear=f.selectedYear=c.getFullYear(),("M"===d||"Y"===d)&&this._notifyChange(f)
},_restrictMinMax:function(c,f){var a=this._getMinMaxDate(c,"min"),d=this._getMinMaxDate(c,"max"),g=a&&a>f?a:f;
return d&&g>d?d:g
},_notifyChange:function(a){var c=this._get(a,"onChangeMonthYear");
c&&c.apply(a.input?a.input[0]:null,[a.selectedYear,a.selectedMonth+1,a])
},_getNumberOfMonths:function(a){var c=this._get(a,"numberOfMonths");
return null==c?[1,1]:"number"==typeof c?[1,c]:c
},_getMinMaxDate:function(a,c){return this._determineDate(a,this._get(a,c+"Date"),null)
},_getDaysInMonth:function(a,c){return 32-this._daylightSavingAdjust(new Date(a,c,32)).getDate()
},_getFirstDayOfMonth:function(a,c){return new Date(a,c,1).getDay()
},_canAdjustMonth:function(c,f,a,d){var h=this._getNumberOfMonths(c),g=this._daylightSavingAdjust(new Date(a,d+(0>f?f:h[0]*h[1]),1));
return 0>f&&g.setDate(this._getDaysInMonth(g.getFullYear(),g.getMonth())),this._isInRange(c,g)
},_isInRange:function(u,m){var g,v,f=this._getMinMaxDate(u,"min"),d=this._getMinMaxDate(u,"max"),p=null,c=null,l=this._get(u,"yearRange");
return l&&(g=l.split(":"),v=(new Date).getFullYear(),p=parseInt(g[0],10),c=parseInt(g[1],10),g[0].match(/[+\-].*/)&&(p+=v),g[1].match(/[+\-].*/)&&(c+=v)),(!f||m.getTime()>=f.getTime())&&(!d||m.getTime()<=d.getTime())&&(!p||m.getFullYear()>=p)&&(!c||c>=m.getFullYear())
},_getFormatConfig:function(a){var c=this._get(a,"shortYearCutoff");
return c="string"!=typeof c?c:(new Date).getFullYear()%100+parseInt(c,10),{shortYearCutoff:c,dayNamesShort:this._get(a,"dayNamesShort"),dayNames:this._get(a,"dayNames"),monthNamesShort:this._get(a,"monthNamesShort"),monthNames:this._get(a,"monthNames")}
},_formatDate:function(c,f,a,d){f||(c.currentDay=c.selectedDay,c.currentMonth=c.selectedMonth,c.currentYear=c.selectedYear);
var g=f?"object"==typeof f?f:this._daylightSavingAdjust(new Date(d,a,f)):this._daylightSavingAdjust(new Date(c.currentYear,c.currentMonth,c.currentDay));
return this.formatDate(this._get(c,"dateFormat"),g,this._getFormatConfig(c))
}}),k.fn.datepicker=function(c){if(!this.length){return this
}k.datepicker.initialized||(k(document).on("mousedown",k.datepicker._checkExternalClick),k.datepicker.initialized=!0),0===k("#"+k.datepicker._mainDivId).length&&k("body").append(k.datepicker.dpDiv);
var a=Array.prototype.slice.call(arguments,1);
return"string"!=typeof c||"isDisabled"!==c&&"getDate"!==c&&"widget"!==c?"option"===c&&2===arguments.length&&"string"==typeof arguments[1]?k.datepicker["_"+c+"Datepicker"].apply(k.datepicker,[this[0]].concat(a)):this.each(function(){"string"==typeof c?k.datepicker["_"+c+"Datepicker"].apply(k.datepicker,[this].concat(a)):k.datepicker._attachDatepicker(this,c)
}):k.datepicker["_"+c+"Datepicker"].apply(k.datepicker,[this[0]].concat(a))
},k.datepicker=new q,k.datepicker.initialized=!1,k.datepicker.uuid=(new Date).getTime(),k.datepicker.version="1.12.1",k.datepicker,k.ui.ie=!!/msie [\w.]+/.exec(navigator.userAgent.toLowerCase());
var K=!1;
k(document).on("mouseup",function(){K=!1
}),k.widget("ui.mouse",{version:"1.12.1",options:{cancel:"input, textarea, button, select, option",distance:1,delay:0},_mouseInit:function(){var a=this;
this.element.on("mousedown."+this.widgetName,function(c){return a._mouseDown(c)
}).on("click."+this.widgetName,function(c){return !0===k.data(c.target,a.widgetName+".preventClickEvent")?(k.removeData(c.target,a.widgetName+".preventClickEvent"),c.stopImmediatePropagation(),!1):void 0
}),this.started=!1
},_mouseDestroy:function(){this.element.off("."+this.widgetName),this._mouseMoveDelegate&&this.document.off("mousemove."+this.widgetName,this._mouseMoveDelegate).off("mouseup."+this.widgetName,this._mouseUpDelegate)
},_mouseDown:function(d){if(!K){this._mouseMoved=!1,this._mouseStarted&&this._mouseUp(d),this._mouseDownEvent=d;
var a=this,c=1===d.which,f="string"==typeof this.options.cancel&&d.target.nodeName?k(d.target).closest(this.options.cancel).length:!1;
return c&&!f&&this._mouseCapture(d)?(this.mouseDelayMet=!this.options.delay,this.mouseDelayMet||(this._mouseDelayTimer=setTimeout(function(){a.mouseDelayMet=!0
},this.options.delay)),this._mouseDistanceMet(d)&&this._mouseDelayMet(d)&&(this._mouseStarted=this._mouseStart(d)!==!1,!this._mouseStarted)?(d.preventDefault(),!0):(!0===k.data(d.target,this.widgetName+".preventClickEvent")&&k.removeData(d.target,this.widgetName+".preventClickEvent"),this._mouseMoveDelegate=function(e){return a._mouseMove(e)
},this._mouseUpDelegate=function(e){return a._mouseUp(e)
},this.document.on("mousemove."+this.widgetName,this._mouseMoveDelegate).on("mouseup."+this.widgetName,this._mouseUpDelegate),d.preventDefault(),K=!0,!0)):!0
}},_mouseMove:function(a){if(this._mouseMoved){if(k.ui.ie&&(!document.documentMode||9>document.documentMode)&&!a.button){return this._mouseUp(a)
}if(!a.which){if(a.originalEvent.altKey||a.originalEvent.ctrlKey||a.originalEvent.metaKey||a.originalEvent.shiftKey){this.ignoreMissingWhich=!0
}else{if(!this.ignoreMissingWhich){return this._mouseUp(a)
}}}}return(a.which||a.button)&&(this._mouseMoved=!0),this._mouseStarted?(this._mouseDrag(a),a.preventDefault()):(this._mouseDistanceMet(a)&&this._mouseDelayMet(a)&&(this._mouseStarted=this._mouseStart(this._mouseDownEvent,a)!==!1,this._mouseStarted?this._mouseDrag(a):this._mouseUp(a)),!this._mouseStarted)
},_mouseUp:function(a){this.document.off("mousemove."+this.widgetName,this._mouseMoveDelegate).off("mouseup."+this.widgetName,this._mouseUpDelegate),this._mouseStarted&&(this._mouseStarted=!1,a.target===this._mouseDownEvent.target&&k.data(a.target,this.widgetName+".preventClickEvent",!0),this._mouseStop(a)),this._mouseDelayTimer&&(clearTimeout(this._mouseDelayTimer),delete this._mouseDelayTimer),this.ignoreMissingWhich=!1,K=!1,a.preventDefault()
},_mouseDistanceMet:function(a){return Math.max(Math.abs(this._mouseDownEvent.pageX-a.pageX),Math.abs(this._mouseDownEvent.pageY-a.pageY))>=this.options.distance
},_mouseDelayMet:function(){return this.mouseDelayMet
},_mouseStart:function(){},_mouseDrag:function(){},_mouseStop:function(){},_mouseCapture:function(){return !0
}}),k.ui.plugin={add:function(d,a,c){var g,f=k.ui[d].prototype;
for(g in c){f.plugins[g]=f.plugins[g]||[],f.plugins[g].push([a,c[g]])
}},call:function(c,f,a,d){var h,g=c.plugins[f];
if(g&&(d||c.element[0].parentNode&&11!==c.element[0].parentNode.nodeType)){for(h=0;
g.length>h;
h++){c.options[g[h][0]]&&g[h][1].apply(c.element,a)
}}}},k.ui.safeBlur=function(a){a&&"body"!==a.nodeName.toLowerCase()&&k(a).trigger("blur")
},k.widget("ui.draggable",k.ui.mouse,{version:"1.12.1",widgetEventPrefix:"drag",options:{addClasses:!0,appendTo:"parent",axis:!1,connectToSortable:!1,containment:!1,cursor:"auto",cursorAt:!1,grid:!1,handle:!1,helper:"original",iframeFix:!1,opacity:!1,refreshPositions:!1,revert:!1,revertDuration:500,scope:"default",scroll:!0,scrollSensitivity:20,scrollSpeed:20,snap:!1,snapMode:"both",snapTolerance:20,stack:!1,zIndex:!1,drag:null,start:null,stop:null},_create:function(){"original"===this.options.helper&&this._setPositionRelative(),this.options.addClasses&&this._addClass("ui-draggable"),this._setHandleClassName(),this._mouseInit()
},_setOption:function(a,c){this._super(a,c),"handle"===a&&(this._removeHandleClassName(),this._setHandleClassName())
},_destroy:function(){return(this.helper||this.element).is(".ui-draggable-dragging")?(this.destroyOnClear=!0,void 0):(this._removeHandleClassName(),this._mouseDestroy(),void 0)
},_mouseCapture:function(c){var a=this.options;
return this.helper||a.disabled||k(c.target).closest(".ui-resizable-handle").length>0?!1:(this.handle=this._getHandle(c),this.handle?(this._blurActiveElement(c),this._blockFrames(a.iframeFix===!0?"iframe":a.iframeFix),!0):!1)
},_blockFrames:function(a){this.iframeBlocks=this.document.find(a).map(function(){var c=k(this);
return k("<div>").css("position","absolute").appendTo(c.parent()).outerWidth(c.outerWidth()).outerHeight(c.outerHeight()).offset(c.offset())[0]
})
},_unblockFrames:function(){this.iframeBlocks&&(this.iframeBlocks.remove(),delete this.iframeBlocks)
},_blurActiveElement:function(d){var a=k.ui.safeActiveElement(this.document[0]),c=k(d.target);
c.closest(a).length||k.ui.safeBlur(a)
},_mouseStart:function(c){var a=this.options;
return this.helper=this._createHelper(c),this._addClass(this.helper,"ui-draggable-dragging"),this._cacheHelperProportions(),k.ui.ddmanager&&(k.ui.ddmanager.current=this),this._cacheMargins(),this.cssPosition=this.helper.css("position"),this.scrollParent=this.helper.scrollParent(!0),this.offsetParent=this.helper.offsetParent(),this.hasFixedAncestor=this.helper.parents().filter(function(){return"fixed"===k(this).css("position")
}).length>0,this.positionAbs=this.element.offset(),this._refreshOffsets(c),this.originalPosition=this.position=this._generatePosition(c,!1),this.originalPageX=c.pageX,this.originalPageY=c.pageY,a.cursorAt&&this._adjustOffsetFromHelper(a.cursorAt),this._setContainment(),this._trigger("start",c)===!1?(this._clear(),!1):(this._cacheHelperProportions(),k.ui.ddmanager&&!a.dropBehaviour&&k.ui.ddmanager.prepareOffsets(this,c),this._mouseDrag(c,!0),k.ui.ddmanager&&k.ui.ddmanager.dragStart(this,c),!0)
},_refreshOffsets:function(a){this.offset={top:this.positionAbs.top-this.margins.top,left:this.positionAbs.left-this.margins.left,scroll:!1,parent:this._getParentOffset(),relative:this._getRelativeOffset()},this.offset.click={left:a.pageX-this.offset.left,top:a.pageY-this.offset.top}
},_mouseDrag:function(d,a){if(this.hasFixedAncestor&&(this.offset.parent=this._getParentOffset()),this.position=this._generatePosition(d,!0),this.positionAbs=this._convertPositionTo("absolute"),!a){var c=this._uiHash();
if(this._trigger("drag",d,c)===!1){return this._mouseUp(new k.Event("mouseup",d)),!1
}this.position=c.position
}return this.helper[0].style.left=this.position.left+"px",this.helper[0].style.top=this.position.top+"px",k.ui.ddmanager&&k.ui.ddmanager.drag(this,d),!1
},_mouseStop:function(d){var a=this,c=!1;
return k.ui.ddmanager&&!this.options.dropBehaviour&&(c=k.ui.ddmanager.drop(this,d)),this.dropped&&(c=this.dropped,this.dropped=!1),"invalid"===this.options.revert&&!c||"valid"===this.options.revert&&c||this.options.revert===!0||k.isFunction(this.options.revert)&&this.options.revert.call(this.element,c)?k(this.helper).animate(this.originalPosition,parseInt(this.options.revertDuration,10),function(){a._trigger("stop",d)!==!1&&a._clear()
}):this._trigger("stop",d)!==!1&&this._clear(),!1
},_mouseUp:function(a){return this._unblockFrames(),k.ui.ddmanager&&k.ui.ddmanager.dragStop(this,a),this.handleElement.is(a.target)&&this.element.trigger("focus"),k.ui.mouse.prototype._mouseUp.call(this,a)
},cancel:function(){return this.helper.is(".ui-draggable-dragging")?this._mouseUp(new k.Event("mouseup",{target:this.element[0]})):this._clear(),this
},_getHandle:function(a){return this.options.handle?!!k(a.target).closest(this.element.find(this.options.handle)).length:!0
},_setHandleClassName:function(){this.handleElement=this.options.handle?this.element.find(this.options.handle):this.element,this._addClass(this.handleElement,"ui-draggable-handle")
},_removeHandleClassName:function(){this._removeClass(this.handleElement,"ui-draggable-handle")
},_createHelper:function(d){var a=this.options,c=k.isFunction(a.helper),f=c?k(a.helper.apply(this.element[0],[d])):"clone"===a.helper?this.element.clone().removeAttr("id"):this.element;
return f.parents("body").length||f.appendTo("parent"===a.appendTo?this.element[0].parentNode:a.appendTo),c&&f[0]===this.element[0]&&this._setPositionRelative(),f[0]===this.element[0]||/(fixed|absolute)/.test(f.css("position"))||f.css("position","absolute"),f
},_setPositionRelative:function(){/^(?:r|a|f)/.test(this.element.css("position"))||(this.element[0].style.position="relative")
},_adjustOffsetFromHelper:function(a){"string"==typeof a&&(a=a.split(" ")),k.isArray(a)&&(a={left:+a[0],top:+a[1]||0}),"left" in a&&(this.offset.click.left=a.left+this.margins.left),"right" in a&&(this.offset.click.left=this.helperProportions.width-a.right+this.margins.left),"top" in a&&(this.offset.click.top=a.top+this.margins.top),"bottom" in a&&(this.offset.click.top=this.helperProportions.height-a.bottom+this.margins.top)
},_isRootNode:function(a){return/(html|body)/i.test(a.tagName)||a===this.document[0]
},_getParentOffset:function(){var c=this.offsetParent.offset(),a=this.document[0];
return"absolute"===this.cssPosition&&this.scrollParent[0]!==a&&k.contains(this.scrollParent[0],this.offsetParent[0])&&(c.left+=this.scrollParent.scrollLeft(),c.top+=this.scrollParent.scrollTop()),this._isRootNode(this.offsetParent[0])&&(c={top:0,left:0}),{top:c.top+(parseInt(this.offsetParent.css("borderTopWidth"),10)||0),left:c.left+(parseInt(this.offsetParent.css("borderLeftWidth"),10)||0)}
},_getRelativeOffset:function(){if("relative"!==this.cssPosition){return{top:0,left:0}
}var a=this.element.position(),c=this._isRootNode(this.scrollParent[0]);
return{top:a.top-(parseInt(this.helper.css("top"),10)||0)+(c?0:this.scrollParent.scrollTop()),left:a.left-(parseInt(this.helper.css("left"),10)||0)+(c?0:this.scrollParent.scrollLeft())}
},_cacheMargins:function(){this.margins={left:parseInt(this.element.css("marginLeft"),10)||0,top:parseInt(this.element.css("marginTop"),10)||0,right:parseInt(this.element.css("marginRight"),10)||0,bottom:parseInt(this.element.css("marginBottom"),10)||0}
},_cacheHelperProportions:function(){this.helperProportions={width:this.helper.outerWidth(),height:this.helper.outerHeight()}
},_setContainment:function(){var d,a,c,g=this.options,f=this.document[0];
return this.relativeContainer=null,g.containment?"window"===g.containment?(this.containment=[k(window).scrollLeft()-this.offset.relative.left-this.offset.parent.left,k(window).scrollTop()-this.offset.relative.top-this.offset.parent.top,k(window).scrollLeft()+k(window).width()-this.helperProportions.width-this.margins.left,k(window).scrollTop()+(k(window).height()||f.body.parentNode.scrollHeight)-this.helperProportions.height-this.margins.top],void 0):"document"===g.containment?(this.containment=[0,0,k(f).width()-this.helperProportions.width-this.margins.left,(k(f).height()||f.body.parentNode.scrollHeight)-this.helperProportions.height-this.margins.top],void 0):g.containment.constructor===Array?(this.containment=g.containment,void 0):("parent"===g.containment&&(g.containment=this.helper[0].parentNode),a=k(g.containment),c=a[0],c&&(d=/(scroll|auto)/.test(a.css("overflow")),this.containment=[(parseInt(a.css("borderLeftWidth"),10)||0)+(parseInt(a.css("paddingLeft"),10)||0),(parseInt(a.css("borderTopWidth"),10)||0)+(parseInt(a.css("paddingTop"),10)||0),(d?Math.max(c.scrollWidth,c.offsetWidth):c.offsetWidth)-(parseInt(a.css("borderRightWidth"),10)||0)-(parseInt(a.css("paddingRight"),10)||0)-this.helperProportions.width-this.margins.left-this.margins.right,(d?Math.max(c.scrollHeight,c.offsetHeight):c.offsetHeight)-(parseInt(a.css("borderBottomWidth"),10)||0)-(parseInt(a.css("paddingBottom"),10)||0)-this.helperProportions.height-this.margins.top-this.margins.bottom],this.relativeContainer=a),void 0):(this.containment=null,void 0)
},_convertPositionTo:function(c,f){f||(f=this.position);
var a="absolute"===c?1:-1,d=this._isRootNode(this.scrollParent[0]);
return{top:f.top+this.offset.relative.top*a+this.offset.parent.top*a-("fixed"===this.cssPosition?-this.offset.scroll.top:d?0:this.offset.scroll.top)*a,left:f.left+this.offset.relative.left*a+this.offset.parent.left*a-("fixed"===this.cssPosition?-this.offset.scroll.left:d?0:this.offset.scroll.left)*a}
},_generatePosition:function(L,u){var m,M,f,d,v=this.options,c=this._isRootNode(this.scrollParent[0]),p=L.pageX,g=L.pageY;
return c&&this.offset.scroll||(this.offset.scroll={top:this.scrollParent.scrollTop(),left:this.scrollParent.scrollLeft()}),u&&(this.containment&&(this.relativeContainer?(M=this.relativeContainer.offset(),m=[this.containment[0]+M.left,this.containment[1]+M.top,this.containment[2]+M.left,this.containment[3]+M.top]):m=this.containment,L.pageX-this.offset.click.left<m[0]&&(p=m[0]+this.offset.click.left),L.pageY-this.offset.click.top<m[1]&&(g=m[1]+this.offset.click.top),L.pageX-this.offset.click.left>m[2]&&(p=m[2]+this.offset.click.left),L.pageY-this.offset.click.top>m[3]&&(g=m[3]+this.offset.click.top)),v.grid&&(f=v.grid[1]?this.originalPageY+Math.round((g-this.originalPageY)/v.grid[1])*v.grid[1]:this.originalPageY,g=m?f-this.offset.click.top>=m[1]||f-this.offset.click.top>m[3]?f:f-this.offset.click.top>=m[1]?f-v.grid[1]:f+v.grid[1]:f,d=v.grid[0]?this.originalPageX+Math.round((p-this.originalPageX)/v.grid[0])*v.grid[0]:this.originalPageX,p=m?d-this.offset.click.left>=m[0]||d-this.offset.click.left>m[2]?d:d-this.offset.click.left>=m[0]?d-v.grid[0]:d+v.grid[0]:d),"y"===v.axis&&(p=this.originalPageX),"x"===v.axis&&(g=this.originalPageY)),{top:g-this.offset.click.top-this.offset.relative.top-this.offset.parent.top+("fixed"===this.cssPosition?-this.offset.scroll.top:c?0:this.offset.scroll.top),left:p-this.offset.click.left-this.offset.relative.left-this.offset.parent.left+("fixed"===this.cssPosition?-this.offset.scroll.left:c?0:this.offset.scroll.left)}
},_clear:function(){this._removeClass(this.helper,"ui-draggable-dragging"),this.helper[0]===this.element[0]||this.cancelHelperRemoval||this.helper.remove(),this.helper=null,this.cancelHelperRemoval=!1,this.destroyOnClear&&this.destroy()
},_trigger:function(d,a,c){return c=c||this._uiHash(),k.ui.plugin.call(this,d,[a,c,this],!0),/^(drag|start|stop)/.test(d)&&(this.positionAbs=this._convertPositionTo("absolute"),c.offset=this.positionAbs),k.Widget.prototype._trigger.call(this,d,a,c)
},plugins:{},_uiHash:function(){return{helper:this.helper,position:this.position,originalPosition:this.originalPosition,offset:this.positionAbs}
}}),k.ui.plugin.add("draggable","connectToSortable",{start:function(d,a,c){var f=k.extend({},a,{item:c.element});
c.sortables=[],k(c.options.connectToSortable).each(function(){var e=k(this).sortable("instance");
e&&!e.options.disabled&&(c.sortables.push(e),e.refreshPositions(),e._trigger("activate",d,f))
})
},stop:function(d,a,c){var f=k.extend({},a,{item:c.element});
c.cancelHelperRemoval=!1,k.each(c.sortables,function(){var e=this;
e.isOver?(e.isOver=0,c.cancelHelperRemoval=!0,e.cancelHelperRemoval=!1,e._storedCSS={position:e.placeholder.css("position"),top:e.placeholder.css("top"),left:e.placeholder.css("left")},e._mouseStop(d),e.options.helper=e.options._helper):(e.cancelHelperRemoval=!0,e._trigger("deactivate",d,f))
})
},drag:function(d,a,c){k.each(c.sortables,function(){var f=!1,e=this;
e.positionAbs=c.positionAbs,e.helperProportions=c.helperProportions,e.offset.click=c.offset.click,e._intersectsWith(e.containerCache)&&(f=!0,k.each(c.sortables,function(){return this.positionAbs=c.positionAbs,this.helperProportions=c.helperProportions,this.offset.click=c.offset.click,this!==e&&this._intersectsWith(this.containerCache)&&k.contains(e.element[0],this.element[0])&&(f=!1),f
})),f?(e.isOver||(e.isOver=1,c._parent=a.helper.parent(),e.currentItem=a.helper.appendTo(e.element).data("ui-sortable-item",!0),e.options._helper=e.options.helper,e.options.helper=function(){return a.helper[0]
},d.target=e.currentItem[0],e._mouseCapture(d,!0),e._mouseStart(d,!0,!0),e.offset.click.top=c.offset.click.top,e.offset.click.left=c.offset.click.left,e.offset.parent.left-=c.offset.parent.left-e.offset.parent.left,e.offset.parent.top-=c.offset.parent.top-e.offset.parent.top,c._trigger("toSortable",d),c.dropped=e.element,k.each(c.sortables,function(){this.refreshPositions()
}),c.currentItem=c.element,e.fromOutside=c),e.currentItem&&(e._mouseDrag(d),a.position=e.position)):e.isOver&&(e.isOver=0,e.cancelHelperRemoval=!0,e.options._revert=e.options.revert,e.options.revert=!1,e._trigger("out",d,e._uiHash(e)),e._mouseStop(d,!0),e.options.revert=e.options._revert,e.options.helper=e.options._helper,e.placeholder&&e.placeholder.remove(),a.helper.appendTo(c._parent),c._refreshOffsets(d),a.position=c._generatePosition(d,!0),c._trigger("fromSortable",d),c.dropped=!1,k.each(c.sortables,function(){this.refreshPositions()
}))
})
}}),k.ui.plugin.add("draggable","cursor",{start:function(d,a,c){var g=k("body"),f=c.options;
g.css("cursor")&&(f._cursor=g.css("cursor")),g.css("cursor",f.cursor)
},stop:function(d,a,c){var f=c.options;
f._cursor&&k("body").css("cursor",f._cursor)
}}),k.ui.plugin.add("draggable","opacity",{start:function(d,a,c){var g=k(a.helper),f=c.options;
g.css("opacity")&&(f._opacity=g.css("opacity")),g.css("opacity",f.opacity)
},stop:function(d,a,c){var f=c.options;
f._opacity&&k(a.helper).css("opacity",f._opacity)
}}),k.ui.plugin.add("draggable","scroll",{start:function(c,d,a){a.scrollParentNotHidden||(a.scrollParentNotHidden=a.helper.scrollParent(!1)),a.scrollParentNotHidden[0]!==a.document[0]&&"HTML"!==a.scrollParentNotHidden[0].tagName&&(a.overflowOffset=a.scrollParentNotHidden.offset())
},drag:function(h,d,f){var m=f.options,l=!1,c=f.scrollParentNotHidden[0],g=f.document[0];
c!==g&&"HTML"!==c.tagName?(m.axis&&"x"===m.axis||(f.overflowOffset.top+c.offsetHeight-h.pageY<m.scrollSensitivity?c.scrollTop=l=c.scrollTop+m.scrollSpeed:h.pageY-f.overflowOffset.top<m.scrollSensitivity&&(c.scrollTop=l=c.scrollTop-m.scrollSpeed)),m.axis&&"y"===m.axis||(f.overflowOffset.left+c.offsetWidth-h.pageX<m.scrollSensitivity?c.scrollLeft=l=c.scrollLeft+m.scrollSpeed:h.pageX-f.overflowOffset.left<m.scrollSensitivity&&(c.scrollLeft=l=c.scrollLeft-m.scrollSpeed))):(m.axis&&"x"===m.axis||(h.pageY-k(g).scrollTop()<m.scrollSensitivity?l=k(g).scrollTop(k(g).scrollTop()-m.scrollSpeed):k(window).height()-(h.pageY-k(g).scrollTop())<m.scrollSensitivity&&(l=k(g).scrollTop(k(g).scrollTop()+m.scrollSpeed))),m.axis&&"y"===m.axis||(h.pageX-k(g).scrollLeft()<m.scrollSensitivity?l=k(g).scrollLeft(k(g).scrollLeft()-m.scrollSpeed):k(window).width()-(h.pageX-k(g).scrollLeft())<m.scrollSensitivity&&(l=k(g).scrollLeft(k(g).scrollLeft()+m.scrollSpeed)))),l!==!1&&k.ui.ddmanager&&!m.dropBehaviour&&k.ui.ddmanager.prepareOffsets(f,h)
}}),k.ui.plugin.add("draggable","snap",{start:function(d,a,c){var f=c.options;
c.snapElements=[],k(f.snap.constructor!==String?f.snap.items||":data(ui-draggable)":f.snap).each(function(){var h=k(this),g=h.offset();
this!==c.element[0]&&c.snapElements.push({item:this,width:h.outerWidth(),height:h.outerHeight(),top:g.top,left:g.left})
})
},drag:function(X,T,M){var Q,P,ab,N,U,S,Z,L,Y,O,W=M.options,V=W.snapTolerance,R=T.offset.left,ac=R+M.helperProportions.width,t=T.offset.top,aa=t+M.helperProportions.height;
for(Y=M.snapElements.length-1;
Y>=0;
Y--){U=M.snapElements[Y].left-M.margins.left,S=U+M.snapElements[Y].width,Z=M.snapElements[Y].top-M.margins.top,L=Z+M.snapElements[Y].height,U-V>ac||R>S+V||Z-V>aa||t>L+V||!k.contains(M.snapElements[Y].item.ownerDocument,M.snapElements[Y].item)?(M.snapElements[Y].snapping&&M.options.snap.release&&M.options.snap.release.call(M.element,X,k.extend(M._uiHash(),{snapItem:M.snapElements[Y].item})),M.snapElements[Y].snapping=!1):("inner"!==W.snapMode&&(Q=V>=Math.abs(Z-aa),P=V>=Math.abs(L-t),ab=V>=Math.abs(U-ac),N=V>=Math.abs(S-R),Q&&(T.position.top=M._convertPositionTo("relative",{top:Z-M.helperProportions.height,left:0}).top),P&&(T.position.top=M._convertPositionTo("relative",{top:L,left:0}).top),ab&&(T.position.left=M._convertPositionTo("relative",{top:0,left:U-M.helperProportions.width}).left),N&&(T.position.left=M._convertPositionTo("relative",{top:0,left:S}).left)),O=Q||P||ab||N,"outer"!==W.snapMode&&(Q=V>=Math.abs(Z-t),P=V>=Math.abs(L-aa),ab=V>=Math.abs(U-R),N=V>=Math.abs(S-ac),Q&&(T.position.top=M._convertPositionTo("relative",{top:Z,left:0}).top),P&&(T.position.top=M._convertPositionTo("relative",{top:L-M.helperProportions.height,left:0}).top),ab&&(T.position.left=M._convertPositionTo("relative",{top:0,left:U}).left),N&&(T.position.left=M._convertPositionTo("relative",{top:0,left:S-M.helperProportions.width}).left)),!M.snapElements[Y].snapping&&(Q||P||ab||N||O)&&M.options.snap.snap&&M.options.snap.snap.call(M.element,X,k.extend(M._uiHash(),{snapItem:M.snapElements[Y].item})),M.snapElements[Y].snapping=Q||P||ab||N||O)
}}}),k.ui.plugin.add("draggable","stack",{start:function(g,d,f){var l,h=f.options,c=k.makeArray(k(h.stack)).sort(function(m,a){return(parseInt(k(m).css("zIndex"),10)||0)-(parseInt(k(a).css("zIndex"),10)||0)
});
c.length&&(l=parseInt(k(c[0]).css("zIndex"),10)||0,k(c).each(function(a){k(this).css("zIndex",l+a)
}),this.css("zIndex",l+c.length))
}}),k.ui.plugin.add("draggable","zIndex",{start:function(d,a,c){var g=k(a.helper),f=c.options;
g.css("zIndex")&&(f._zIndex=g.css("zIndex")),g.css("zIndex",f.zIndex)
},stop:function(d,a,c){var f=c.options;
f._zIndex&&k(a.helper).css("zIndex",f._zIndex)
}}),k.ui.draggable,k.widget("ui.resizable",k.ui.mouse,{version:"1.12.1",widgetEventPrefix:"resize",options:{alsoResize:!1,animate:!1,animateDuration:"slow",animateEasing:"swing",aspectRatio:!1,autoHide:!1,classes:{"ui-resizable-se":"ui-icon ui-icon-gripsmall-diagonal-se"},containment:!1,ghost:!1,grid:!1,handles:"e,s,se",helper:!1,maxHeight:null,maxWidth:null,minHeight:10,minWidth:10,zIndex:90,resize:null,start:null,stop:null},_num:function(a){return parseFloat(a)||0
},_isNumber:function(a){return !isNaN(parseFloat(a))
},_hasScroll:function(d,a){if("hidden"===k(d).css("overflow")){return !1
}var c=a&&"left"===a?"scrollLeft":"scrollTop",f=!1;
return d[c]>0?!0:(d[c]=1,f=d[c]>0,d[c]=0,f)
},_create:function(){var d,a=this.options,c=this;
this._addClass("ui-resizable"),k.extend(this,{_aspectRatio:!!a.aspectRatio,aspectRatio:a.aspectRatio,originalElement:this.element,_proportionallyResizeElements:[],_helper:a.helper||a.ghost||a.animate?a.helper||"ui-resizable-helper":null}),this.element[0].nodeName.match(/^(canvas|textarea|input|select|button|img)$/i)&&(this.element.wrap(k("<div class='ui-wrapper' style='overflow: hidden;'></div>").css({position:this.element.css("position"),width:this.element.outerWidth(),height:this.element.outerHeight(),top:this.element.css("top"),left:this.element.css("left")})),this.element=this.element.parent().data("ui-resizable",this.element.resizable("instance")),this.elementIsWrapper=!0,d={marginTop:this.originalElement.css("marginTop"),marginRight:this.originalElement.css("marginRight"),marginBottom:this.originalElement.css("marginBottom"),marginLeft:this.originalElement.css("marginLeft")},this.element.css(d),this.originalElement.css("margin",0),this.originalResizeStyle=this.originalElement.css("resize"),this.originalElement.css("resize","none"),this._proportionallyResizeElements.push(this.originalElement.css({position:"static",zoom:1,display:"block"})),this.originalElement.css(d),this._proportionallyResize()),this._setupHandles(),a.autoHide&&k(this.element).on("mouseenter",function(){a.disabled||(c._removeClass("ui-resizable-autohide"),c._handles.show())
}).on("mouseleave",function(){a.disabled||c.resizing||(c._addClass("ui-resizable-autohide"),c._handles.hide())
}),this._mouseInit()
},_destroy:function(){this._mouseDestroy();
var c,a=function(d){k(d).removeData("resizable").removeData("ui-resizable").off(".resizable").find(".ui-resizable-handle").remove()
};
return this.elementIsWrapper&&(a(this.element),c=this.element,this.originalElement.css({position:c.css("position"),width:c.outerWidth(),height:c.outerHeight(),top:c.css("top"),left:c.css("left")}).insertAfter(c),c.remove()),this.originalElement.css("resize",this.originalResizeStyle),a(this.originalElement),this
},_setOption:function(a,c){switch(this._super(a,c),a){case"handles":this._removeHandles(),this._setupHandles();
break;
default:}},_setupHandles:function(){var h,d,f,m,l,c=this.options,g=this;
if(this.handles=c.handles||(k(".ui-resizable-handle",this.element).length?{n:".ui-resizable-n",e:".ui-resizable-e",s:".ui-resizable-s",w:".ui-resizable-w",se:".ui-resizable-se",sw:".ui-resizable-sw",ne:".ui-resizable-ne",nw:".ui-resizable-nw"}:"e,s,se"),this._handles=k(),this.handles.constructor===String){for("all"===this.handles&&(this.handles="n,e,s,w,se,sw,ne,nw"),f=this.handles.split(","),this.handles={},d=0;
f.length>d;
d++){h=k.trim(f[d]),m="ui-resizable-"+h,l=k("<div>"),this._addClass(l,"ui-resizable-handle "+m),l.css({zIndex:c.zIndex}),this.handles[h]=".ui-resizable-"+h,this.element.append(l)
}}this._renderAxis=function(r){var a,p,u,t;
r=r||this.element;
for(a in this.handles){this.handles[a].constructor===String?this.handles[a]=this.element.children(this.handles[a]).first().show():(this.handles[a].jquery||this.handles[a].nodeType)&&(this.handles[a]=k(this.handles[a]),this._on(this.handles[a],{mousedown:g._mouseDown})),this.elementIsWrapper&&this.originalElement[0].nodeName.match(/^(textarea|input|select|button)$/i)&&(p=k(this.handles[a],this.element),t=/sw|ne|nw|se|n|s/.test(a)?p.outerHeight():p.outerWidth(),u=["padding",/ne|nw|n/.test(a)?"Top":/se|sw|s/.test(a)?"Bottom":/^e$/.test(a)?"Right":"Left"].join(""),r.css(u,t),this._proportionallyResize()),this._handles=this._handles.add(this.handles[a])
}},this._renderAxis(this.element),this._handles=this._handles.add(this.element.find(".ui-resizable-handle")),this._handles.disableSelection(),this._handles.on("mouseover",function(){g.resizing||(this.className&&(l=this.className.match(/ui-resizable-(se|sw|ne|nw|n|e|s|w)/i)),g.axis=l&&l[1]?l[1]:"se")
}),c.autoHide&&(this._handles.hide(),this._addClass("ui-resizable-autohide"))
},_removeHandles:function(){this._handles.remove()
},_mouseCapture:function(d){var a,c,f=!1;
for(a in this.handles){c=k(this.handles[a])[0],(c===d.target||k.contains(c,d.target))&&(f=!0)
}return !this.options.disabled&&f
},_mouseStart:function(g){var d,f,l,h=this.options,c=this.element;
return this.resizing=!0,this._renderProxy(),d=this._num(this.helper.css("left")),f=this._num(this.helper.css("top")),h.containment&&(d+=k(h.containment).scrollLeft()||0,f+=k(h.containment).scrollTop()||0),this.offset=this.helper.offset(),this.position={left:d,top:f},this.size=this._helper?{width:this.helper.width(),height:this.helper.height()}:{width:c.width(),height:c.height()},this.originalSize=this._helper?{width:c.outerWidth(),height:c.outerHeight()}:{width:c.width(),height:c.height()},this.sizeDiff={width:c.outerWidth()-c.width(),height:c.outerHeight()-c.height()},this.originalPosition={left:d,top:f},this.originalMousePosition={left:g.pageX,top:g.pageY},this.aspectRatio="number"==typeof h.aspectRatio?h.aspectRatio:this.originalSize.width/this.originalSize.height||1,l=k(".ui-resizable-"+this.axis).css("cursor"),k("body").css("cursor","auto"===l?this.axis+"-resize":l),this._addClass("ui-resizable-resizing"),this._propagate("start",g),!0
},_mouseDrag:function(m){var d,g,t=this.originalMousePosition,p=this.axis,c=m.pageX-t.left||0,l=m.pageY-t.top||0,f=this._change[p];
return this._updatePrevProperties(),f?(d=f.apply(this,[m,c,l]),this._updateVirtualBoundaries(m.shiftKey),(this._aspectRatio||m.shiftKey)&&(d=this._updateRatio(d,m)),d=this._respectSize(d,m),this._updateCache(d),this._propagate("resize",m),g=this._applyChanges(),!this._helper&&this._proportionallyResizeElements.length&&this._proportionallyResize(),k.isEmptyObject(g)||(this._updatePrevProperties(),this._trigger("resize",m,this.ui()),this._applyChanges()),!1):!1
},_mouseStop:function(u){this.resizing=!1;
var p,M,g,f,L,d,t,m=this.options,v=this;
return this._helper&&(p=this._proportionallyResizeElements,M=p.length&&/textarea/i.test(p[0].nodeName),g=M&&this._hasScroll(p[0],"left")?0:v.sizeDiff.height,f=M?0:v.sizeDiff.width,L={width:v.helper.width()-f,height:v.helper.height()-g},d=parseFloat(v.element.css("left"))+(v.position.left-v.originalPosition.left)||null,t=parseFloat(v.element.css("top"))+(v.position.top-v.originalPosition.top)||null,m.animate||this.element.css(k.extend(L,{top:t,left:d})),v.helper.height(v.size.height),v.helper.width(v.size.width),this._helper&&!m.animate&&this._proportionallyResize()),k("body").css("cursor","auto"),this._removeClass("ui-resizable-resizing"),this._propagate("stop",u),this._helper&&this.helper.remove(),!1
},_updatePrevProperties:function(){this.prevPosition={top:this.position.top,left:this.position.left},this.prevSize={width:this.size.width,height:this.size.height}
},_applyChanges:function(){var a={};
return this.position.top!==this.prevPosition.top&&(a.top=this.position.top+"px"),this.position.left!==this.prevPosition.left&&(a.left=this.position.left+"px"),this.size.width!==this.prevSize.width&&(a.width=this.size.width+"px"),this.size.height!==this.prevSize.height&&(a.height=this.size.height+"px"),this.helper.css(a),a
},_updateVirtualBoundaries:function(f){var h,d,g,m,l,c=this.options;
l={minWidth:this._isNumber(c.minWidth)?c.minWidth:0,maxWidth:this._isNumber(c.maxWidth)?c.maxWidth:1/0,minHeight:this._isNumber(c.minHeight)?c.minHeight:0,maxHeight:this._isNumber(c.maxHeight)?c.maxHeight:1/0},(this._aspectRatio||f)&&(h=l.minHeight*this.aspectRatio,g=l.minWidth/this.aspectRatio,d=l.maxHeight*this.aspectRatio,m=l.maxWidth/this.aspectRatio,h>l.minWidth&&(l.minWidth=h),g>l.minHeight&&(l.minHeight=g),l.maxWidth>d&&(l.maxWidth=d),l.maxHeight>m&&(l.maxHeight=m)),this._vBoundaries=l
},_updateCache:function(a){this.offset=this.helper.offset(),this._isNumber(a.left)&&(this.position.left=a.left),this._isNumber(a.top)&&(this.position.top=a.top),this._isNumber(a.height)&&(this.size.height=a.height),this._isNumber(a.width)&&(this.size.width=a.width)
},_updateRatio:function(c){var f=this.position,a=this.size,d=this.axis;
return this._isNumber(c.height)?c.width=c.height*this.aspectRatio:this._isNumber(c.width)&&(c.height=c.width/this.aspectRatio),"sw"===d&&(c.left=f.left+(a.width-c.width),c.top=null),"nw"===d&&(c.top=f.top+(a.height-c.height),c.left=f.left+(a.width-c.width)),c
},_respectSize:function(N){var v=this._vBoundaries,p=this.axis,O=this._isNumber(N.width)&&v.maxWidth&&v.maxWidth<N.width,g=this._isNumber(N.height)&&v.maxHeight&&v.maxHeight<N.height,f=this._isNumber(N.width)&&v.minWidth&&v.minWidth>N.width,M=this._isNumber(N.height)&&v.minHeight&&v.minHeight>N.height,d=this.originalPosition.left+this.originalSize.width,u=this.originalPosition.top+this.originalSize.height,m=/sw|nw|w/.test(p),L=/nw|ne|n/.test(p);
return f&&(N.width=v.minWidth),M&&(N.height=v.minHeight),O&&(N.width=v.maxWidth),g&&(N.height=v.maxHeight),f&&m&&(N.left=d-v.minWidth),O&&m&&(N.left=d-v.maxWidth),M&&L&&(N.top=u-v.minHeight),g&&L&&(N.top=u-v.maxHeight),N.width||N.height||N.left||!N.top?N.width||N.height||N.top||!N.left||(N.left=null):N.top=null,N
},_getPaddingPlusBorderDimensions:function(c){for(var f=0,a=[],d=[c.css("borderTopWidth"),c.css("borderRightWidth"),c.css("borderBottomWidth"),c.css("borderLeftWidth")],g=[c.css("paddingTop"),c.css("paddingRight"),c.css("paddingBottom"),c.css("paddingLeft")];
4>f;
f++){a[f]=parseFloat(d[f])||0,a[f]+=parseFloat(g[f])||0
}return{height:a[0]+a[2],width:a[1]+a[3]}
},_proportionallyResize:function(){if(this._proportionallyResizeElements.length){for(var c,d=0,a=this.helper||this.element;
this._proportionallyResizeElements.length>d;
d++){c=this._proportionallyResizeElements[d],this.outerDimensions||(this.outerDimensions=this._getPaddingPlusBorderDimensions(c)),c.css({height:a.height()-this.outerDimensions.height||0,width:a.width()-this.outerDimensions.width||0})
}}},_renderProxy:function(){var c=this.element,a=this.options;
this.elementOffset=c.offset(),this._helper?(this.helper=this.helper||k("<div style='overflow:hidden;'></div>"),this._addClass(this.helper,this._helper),this.helper.css({width:this.element.outerWidth(),height:this.element.outerHeight(),position:"absolute",left:this.elementOffset.left+"px",top:this.elementOffset.top+"px",zIndex:++a.zIndex}),this.helper.appendTo("body").disableSelection()):this.helper=this.element
},_change:{e:function(a,c){return{width:this.originalSize.width+c}
},w:function(c,f){var a=this.originalSize,d=this.originalPosition;
return{left:d.left+f,width:a.width-f}
},n:function(c,f,a){var d=this.originalSize,g=this.originalPosition;
return{top:g.top+a,height:d.height-a}
},s:function(c,d,a){return{height:this.originalSize.height+a}
},se:function(d,a,c){return k.extend(this._change.s.apply(this,arguments),this._change.e.apply(this,[d,a,c]))
},sw:function(d,a,c){return k.extend(this._change.s.apply(this,arguments),this._change.w.apply(this,[d,a,c]))
},ne:function(d,a,c){return k.extend(this._change.n.apply(this,arguments),this._change.e.apply(this,[d,a,c]))
},nw:function(d,a,c){return k.extend(this._change.n.apply(this,arguments),this._change.w.apply(this,[d,a,c]))
}},_propagate:function(c,a){k.ui.plugin.call(this,c,[a,this.ui()]),"resize"!==c&&this._trigger(c,a,this.ui())
},plugins:{},ui:function(){return{originalElement:this.originalElement,element:this.element,helper:this.helper,position:this.position,size:this.size,originalSize:this.originalSize,originalPosition:this.originalPosition}
}}),k.ui.plugin.add("resizable","animate",{stop:function(u){var p=k(this).resizable("instance"),M=p.options,g=p._proportionallyResizeElements,f=g.length&&/textarea/i.test(g[0].nodeName),L=f&&p._hasScroll(g[0],"left")?0:p.sizeDiff.height,d=f?0:p.sizeDiff.width,t={width:p.size.width-d,height:p.size.height-L},m=parseFloat(p.element.css("left"))+(p.position.left-p.originalPosition.left)||null,v=parseFloat(p.element.css("top"))+(p.position.top-p.originalPosition.top)||null;
p.element.animate(k.extend(t,v&&m?{top:v,left:m}:{}),{duration:M.animateDuration,easing:M.animateEasing,step:function(){var a={width:parseFloat(p.element.css("width")),height:parseFloat(p.element.css("height")),top:parseFloat(p.element.css("top")),left:parseFloat(p.element.css("left"))};
g&&g.length&&k(g[0]).css({width:a.width,height:a.height}),p._updateCache(a),p._propagate("resize",u)
}})
}}),k.ui.plugin.add("resizable","containment",{start:function(){var L,t,Q,m,g,O,f,v=k(this).resizable("instance"),p=v.options,N=v.element,P=p.containment,M=P instanceof k?P.get(0):/parent/.test(P)?N.parent().get(0):P;
M&&(v.containerElement=k(M),/document/.test(P)||P===document?(v.containerOffset={left:0,top:0},v.containerPosition={left:0,top:0},v.parentData={element:k(document),left:0,top:0,width:k(document).width(),height:k(document).height()||document.body.parentNode.scrollHeight}):(L=k(M),t=[],k(["Top","Right","Left","Bottom"]).each(function(a,c){t[a]=v._num(L.css("padding"+c))
}),v.containerOffset=L.offset(),v.containerPosition=L.position(),v.containerSize={height:L.innerHeight()-t[3],width:L.innerWidth()-t[1]},Q=v.containerOffset,m=v.containerSize.height,g=v.containerSize.width,O=v._hasScroll(M,"left")?M.scrollWidth:g,f=v._hasScroll(M)?M.scrollHeight:m,v.parentData={element:M,left:Q.left,top:Q.top,width:O,height:f}))
},resize:function(N){var L,S,t,m,Q=k(this).resizable("instance"),f=Q.options,M=Q.containerOffset,v=Q.position,P=Q._aspectRatio||N.shiftKey,R={top:0,left:0},O=Q.containerElement,g=!0;
O[0]!==document&&/static/.test(O.css("position"))&&(R=M),v.left<(Q._helper?M.left:0)&&(Q.size.width=Q.size.width+(Q._helper?Q.position.left-M.left:Q.position.left-R.left),P&&(Q.size.height=Q.size.width/Q.aspectRatio,g=!1),Q.position.left=f.helper?M.left:0),v.top<(Q._helper?M.top:0)&&(Q.size.height=Q.size.height+(Q._helper?Q.position.top-M.top:Q.position.top),P&&(Q.size.width=Q.size.height*Q.aspectRatio,g=!1),Q.position.top=Q._helper?M.top:0),t=Q.containerElement.get(0)===Q.element.parent().get(0),m=/relative|absolute/.test(Q.containerElement.css("position")),t&&m?(Q.offset.left=Q.parentData.left+Q.position.left,Q.offset.top=Q.parentData.top+Q.position.top):(Q.offset.left=Q.element.offset().left,Q.offset.top=Q.element.offset().top),L=Math.abs(Q.sizeDiff.width+(Q._helper?Q.offset.left-R.left:Q.offset.left-M.left)),S=Math.abs(Q.sizeDiff.height+(Q._helper?Q.offset.top-R.top:Q.offset.top-M.top)),L+Q.size.width>=Q.parentData.width&&(Q.size.width=Q.parentData.width-L,P&&(Q.size.height=Q.size.width/Q.aspectRatio,g=!1)),S+Q.size.height>=Q.parentData.height&&(Q.size.height=Q.parentData.height-S,P&&(Q.size.width=Q.size.height*Q.aspectRatio,g=!1)),g||(Q.position.left=Q.prevPosition.left,Q.position.top=Q.prevPosition.top,Q.size.width=Q.prevSize.width,Q.size.height=Q.prevSize.height)
},stop:function(){var t=k(this).resizable("instance"),m=t.options,v=t.containerOffset,f=t.containerPosition,d=t.containerElement,u=k(t.helper),c=u.offset(),p=u.outerWidth()-t.sizeDiff.width,g=u.outerHeight()-t.sizeDiff.height;
t._helper&&!m.animate&&/relative/.test(d.css("position"))&&k(this).css({left:c.left-f.left-v.left,width:p,height:g}),t._helper&&!m.animate&&/static/.test(d.css("position"))&&k(this).css({left:c.left-f.left-v.left,width:p,height:g})
}}),k.ui.plugin.add("resizable","alsoResize",{start:function(){var c=k(this).resizable("instance"),a=c.options;
k(a.alsoResize).each(function(){var d=k(this);
d.data("ui-resizable-alsoresize",{width:parseFloat(d.width()),height:parseFloat(d.height()),left:parseFloat(d.css("left")),top:parseFloat(d.css("top"))})
})
},resize:function(h,d){var f=k(this).resizable("instance"),m=f.options,l=f.originalSize,c=f.originalPosition,g={height:f.size.height-l.height||0,width:f.size.width-l.width||0,top:f.position.top-c.top||0,left:f.position.left-c.left||0};
k(m.alsoResize).each(function(){var i=k(this),a=k(this).data("ui-resizable-alsoresize"),r={},p=i.parents(d.originalElement[0]).length?["width","height"]:["width","height","top","left"];
k.each(p,function(o,s){var n=(a[s]||0)+(g[s]||0);
n&&n>=0&&(r[s]=n||null)
}),i.css(r)
})
},stop:function(){k(this).removeData("ui-resizable-alsoresize")
}}),k.ui.plugin.add("resizable","ghost",{start:function(){var c=k(this).resizable("instance"),a=c.size;
c.ghost=c.originalElement.clone(),c.ghost.css({opacity:0.25,display:"block",position:"relative",height:a.height,width:a.width,margin:0,left:0,top:0}),c._addClass(c.ghost,"ui-resizable-ghost"),k.uiBackCompat!==!1&&"string"==typeof c.options.ghost&&c.ghost.addClass(this.options.ghost),c.ghost.appendTo(c.helper)
},resize:function(){var a=k(this).resizable("instance");
a.ghost&&a.ghost.css({position:"relative",height:a.size.height,width:a.size.width})
},stop:function(){var a=k(this).resizable("instance");
a.ghost&&a.helper&&a.helper.get(0).removeChild(a.ghost.get(0))
}}),k.ui.plugin.add("resizable","grid",{resize:function(){var X,T=k(this).resizable("instance"),M=T.options,Q=T.size,P=T.originalSize,aa=T.originalPosition,N=T.axis,U="number"==typeof M.grid?[M.grid,M.grid]:M.grid,S=U[0]||1,Z=U[1]||1,L=Math.round((Q.width-P.width)/S)*S,Y=Math.round((Q.height-P.height)/Z)*Z,O=P.width+L,W=P.height+Y,V=M.maxWidth&&O>M.maxWidth,R=M.maxHeight&&W>M.maxHeight,ab=M.minWidth&&M.minWidth>O,t=M.minHeight&&M.minHeight>W;
M.grid=U,ab&&(O+=S),t&&(W+=Z),V&&(O-=S),R&&(W-=Z),/^(se|s|e)$/.test(N)?(T.size.width=O,T.size.height=W):/^(ne)$/.test(N)?(T.size.width=O,T.size.height=W,T.position.top=aa.top-Y):/^(sw)$/.test(N)?(T.size.width=O,T.size.height=W,T.position.left=aa.left-L):((0>=W-Z||0>=O-S)&&(X=T._getPaddingPlusBorderDimensions(this)),W-Z>0?(T.size.height=W,T.position.top=aa.top-Y):(W=Z-X.height,T.size.height=W,T.position.top=aa.top+P.height-W),O-S>0?(T.size.width=O,T.position.left=aa.left-L):(O=S-X.width,T.size.width=O,T.position.left=aa.left+P.width-O))
}}),k.ui.resizable,k.widget("ui.dialog",{version:"1.12.1",options:{appendTo:"body",autoOpen:!0,buttons:[],classes:{"ui-dialog":"ui-corner-all","ui-dialog-titlebar":"ui-corner-all"},closeOnEscape:!0,closeText:"Close",draggable:!0,hide:null,height:"auto",maxHeight:null,maxWidth:null,minHeight:150,minWidth:150,modal:!1,position:{my:"center",at:"center",of:window,collision:"fit",using:function(c){var a=k(this).css(c).offset().top;
0>a&&k(this).css("top",c.top-a)
}},resizable:!0,show:null,title:null,width:300,beforeClose:null,close:null,drag:null,dragStart:null,dragStop:null,focus:null,open:null,resize:null,resizeStart:null,resizeStop:null},sizeRelatedOptions:{buttons:!0,height:!0,maxHeight:!0,maxWidth:!0,minHeight:!0,minWidth:!0,width:!0},resizableRelatedOptions:{maxHeight:!0,maxWidth:!0,minHeight:!0,minWidth:!0},_create:function(){this.originalCss={display:this.element[0].style.display,width:this.element[0].style.width,minHeight:this.element[0].style.minHeight,maxHeight:this.element[0].style.maxHeight,height:this.element[0].style.height},this.originalPosition={parent:this.element.parent(),index:this.element.parent().children().index(this.element)},this.originalTitle=this.element.attr("title"),null==this.options.title&&null!=this.originalTitle&&(this.options.title=this.originalTitle),this.options.disabled&&(this.options.disabled=!1),this._createWrapper(),this.element.show().removeAttr("title").appendTo(this.uiDialog),this._addClass("ui-dialog-content","ui-widget-content"),this._createTitlebar(),this._createButtonPane(),this.options.draggable&&k.fn.draggable&&this._makeDraggable(),this.options.resizable&&k.fn.resizable&&this._makeResizable(),this._isOpen=!1,this._trackFocus()
},_init:function(){this.options.autoOpen&&this.open()
},_appendTo:function(){var a=this.options.appendTo;
return a&&(a.jquery||a.nodeType)?k(a):this.document.find(a||"body").eq(0)
},_destroy:function(){var a,c=this.originalPosition;
this._untrackInstance(),this._destroyOverlay(),this.element.removeUniqueId().css(this.originalCss).detach(),this.uiDialog.remove(),this.originalTitle&&this.element.attr("title",this.originalTitle),a=c.parent.children().eq(c.index),a.length&&a[0]!==this.element[0]?a.before(this.element):c.parent.append(this.element)
},widget:function(){return this.uiDialog
},disable:k.noop,enable:k.noop,close:function(c){var a=this;
this._isOpen&&this._trigger("beforeClose",c)!==!1&&(this._isOpen=!1,this._focusedElement=null,this._destroyOverlay(),this._untrackInstance(),this.opener.filter(":focusable").trigger("focus").length||k.ui.safeBlur(k.ui.safeActiveElement(this.document[0])),this._hide(this.uiDialog,this.options.hide,function(){a._trigger("close",c)
}))
},isOpen:function(){return this._isOpen
},moveToTop:function(){this._moveToTop()
},_moveToTop:function(d,a){var c=!1,g=this.uiDialog.siblings(".ui-front:visible").map(function(){return +k(this).css("z-index")
}).get(),f=Math.max.apply(null,g);
return f>=+this.uiDialog.css("z-index")&&(this.uiDialog.css("z-index",f+1),c=!0),c&&!a&&this._trigger("focus",d),c
},open:function(){var a=this;
return this._isOpen?(this._moveToTop()&&this._focusTabbable(),void 0):(this._isOpen=!0,this.opener=k(k.ui.safeActiveElement(this.document[0])),this._size(),this._position(),this._createOverlay(),this._moveToTop(null,!0),this.overlay&&this.overlay.css("z-index",this.uiDialog.css("z-index")-1),this._show(this.uiDialog,this.options.show,function(){a._focusTabbable(),a._trigger("focus")
}),this._makeFocusTarget(),this._trigger("open"),void 0)
},_focusTabbable:function(){var a=this._focusedElement;
a||(a=this.element.find("[autofocus]")),a.length||(a=this.element.find(":tabbable")),a.length||(a=this.uiDialogButtonPane.find(":tabbable")),a.length||(a=this.uiDialogTitlebarClose.filter(":tabbable")),a.length||(a=this.uiDialog),a.eq(0).trigger("focus")
},_keepFocus:function(c){function a(){var f=k.ui.safeActiveElement(this.document[0]),d=this.uiDialog[0]===f||k.contains(this.uiDialog[0],f);
d||this._focusTabbable()
}c.preventDefault(),a.call(this),this._delay(a)
},_createWrapper:function(){this.uiDialog=k("<div>").hide().attr({tabIndex:-1,role:"dialog"}).appendTo(this._appendTo()),this._addClass(this.uiDialog,"ui-dialog","ui-widget ui-widget-content ui-front"),this._on(this.uiDialog,{keydown:function(d){if(this.options.closeOnEscape&&!d.isDefaultPrevented()&&d.keyCode&&d.keyCode===k.ui.keyCode.ESCAPE){return d.preventDefault(),this.close(d),void 0
}if(d.keyCode===k.ui.keyCode.TAB&&!d.isDefaultPrevented()){var a=this.uiDialog.find(":tabbable"),c=a.filter(":first"),f=a.filter(":last");
d.target!==f[0]&&d.target!==this.uiDialog[0]||d.shiftKey?d.target!==c[0]&&d.target!==this.uiDialog[0]||!d.shiftKey||(this._delay(function(){f.trigger("focus")
}),d.preventDefault()):(this._delay(function(){c.trigger("focus")
}),d.preventDefault())
}},mousedown:function(a){this._moveToTop(a)&&this._focusTabbable()
}}),this.element.find("[aria-describedby]").length||this.uiDialog.attr({"aria-describedby":this.element.uniqueId().attr("id")})
},_createTitlebar:function(){var a;
this.uiDialogTitlebar=k("<div>"),this._addClass(this.uiDialogTitlebar,"ui-dialog-titlebar","ui-widget-header ui-helper-clearfix"),this._on(this.uiDialogTitlebar,{mousedown:function(c){k(c.target).closest(".ui-dialog-titlebar-close")||this.uiDialog.trigger("focus")
}}),this.uiDialogTitlebarClose=k("<button type='button'></button>").button({label:k("<a>").text(this.options.closeText).html(),icon:"ui-icon-closethick",showLabel:!1}).appendTo(this.uiDialogTitlebar),this._addClass(this.uiDialogTitlebarClose,"ui-dialog-titlebar-close"),this._on(this.uiDialogTitlebarClose,{click:function(c){c.preventDefault(),this.close(c)
}}),a=k("<span>").uniqueId().prependTo(this.uiDialogTitlebar),this._addClass(a,"ui-dialog-title"),this._title(a),this.uiDialogTitlebar.prependTo(this.uiDialog),this.uiDialog.attr({"aria-labelledby":a.attr("id")})
},_title:function(a){this.options.title?a.text(this.options.title):a.html("&#160;")
},_createButtonPane:function(){this.uiDialogButtonPane=k("<div>"),this._addClass(this.uiDialogButtonPane,"ui-dialog-buttonpane","ui-widget-content ui-helper-clearfix"),this.uiButtonSet=k("<div>").appendTo(this.uiDialogButtonPane),this._addClass(this.uiButtonSet,"ui-dialog-buttonset"),this._createButtons()
},_createButtons:function(){var c=this,a=this.options.buttons;
return this.uiDialogButtonPane.remove(),this.uiButtonSet.empty(),k.isEmptyObject(a)||k.isArray(a)&&!a.length?(this._removeClass(this.uiDialog,"ui-dialog-buttons"),void 0):(k.each(a,function(d,e){var g,f;
e=k.isFunction(e)?{click:e,text:d}:e,e=k.extend({type:"button"},e),g=e.click,f={icon:e.icon,iconPosition:e.iconPosition,showLabel:e.showLabel,icons:e.icons,text:e.text},delete e.click,delete e.icon,delete e.iconPosition,delete e.showLabel,delete e.icons,"boolean"==typeof e.text&&delete e.text,k("<button></button>",e).button(f).appendTo(c.uiButtonSet).on("click",function(){g.apply(c.element[0],arguments)
})
}),this._addClass(this.uiDialog,"ui-dialog-buttons"),this.uiDialogButtonPane.appendTo(this.uiDialog),void 0)
},_makeDraggable:function(){function d(e){return{position:e.position,offset:e.offset}
}var a=this,c=this.options;
this.uiDialog.draggable({cancel:".ui-dialog-content, .ui-dialog-titlebar-close",handle:".ui-dialog-titlebar",containment:"document",start:function(e,f){a._addClass(k(this),"ui-dialog-dragging"),a._blockFrames(),a._trigger("dragStart",e,d(f))
},drag:function(e,f){a._trigger("drag",e,d(f))
},stop:function(h,g){var e=g.offset.left-a.document.scrollLeft(),f=g.offset.top-a.document.scrollTop();
c.position={my:"left top",at:"left"+(e>=0?"+":"")+e+" top"+(f>=0?"+":"")+f,of:a.window},a._removeClass(k(this),"ui-dialog-dragging"),a._unblockFrames(),a._trigger("dragStop",h,d(g))
}})
},_makeResizable:function(){function g(a){return{originalPosition:a.originalPosition,originalSize:a.originalSize,position:a.position,size:a.size}
}var d=this,f=this.options,l=f.resizable,h=this.uiDialog.css("position"),c="string"==typeof l?l:"n,e,s,w,se,sw,ne,nw";
this.uiDialog.resizable({cancel:".ui-dialog-content",containment:"document",alsoResize:this.element,maxWidth:f.maxWidth,maxHeight:f.maxHeight,minWidth:f.minWidth,minHeight:this._minHeight(),handles:c,start:function(a,e){d._addClass(k(this),"ui-dialog-resizing"),d._blockFrames(),d._trigger("resizeStart",a,g(e))
},resize:function(a,e){d._trigger("resize",a,g(e))
},stop:function(s,p){var e=d.uiDialog.offset(),m=e.left-d.document.scrollLeft(),i=e.top-d.document.scrollTop();
f.height=d.uiDialog.height(),f.width=d.uiDialog.width(),f.position={my:"left top",at:"left"+(m>=0?"+":"")+m+" top"+(i>=0?"+":"")+i,of:d.window},d._removeClass(k(this),"ui-dialog-resizing"),d._unblockFrames(),d._trigger("resizeStop",s,g(p))
}}).css("position",h)
},_trackFocus:function(){this._on(this.widget(),{focusin:function(a){this._makeFocusTarget(),this._focusedElement=k(a.target)
}})
},_makeFocusTarget:function(){this._untrackInstance(),this._trackingInstances().unshift(this)
},_untrackInstance:function(){var c=this._trackingInstances(),a=k.inArray(this,c);
-1!==a&&c.splice(a,1)
},_trackingInstances:function(){var a=this.document.data("ui-dialog-instances");
return a||(a=[],this.document.data("ui-dialog-instances",a)),a
},_minHeight:function(){var a=this.options;
return"auto"===a.height?a.minHeight:Math.min(a.minHeight,a.height)
},_position:function(){var a=this.uiDialog.is(":visible");
a||this.uiDialog.show(),this.uiDialog.position(this.options.position),a||this.uiDialog.hide()
},_setOptions:function(d){var a=this,c=!1,f={};
k.each(d,function(g,h){a._setOption(g,h),g in a.sizeRelatedOptions&&(c=!0),g in a.resizableRelatedOptions&&(f[g]=h)
}),c&&(this._size(),this._position()),this.uiDialog.is(":data(ui-resizable)")&&this.uiDialog.resizable("option",f)
},_setOption:function(d,a){var c,g,f=this.uiDialog;
"disabled"!==d&&(this._super(d,a),"appendTo"===d&&this.uiDialog.appendTo(this._appendTo()),"buttons"===d&&this._createButtons(),"closeText"===d&&this.uiDialogTitlebarClose.button({label:k("<a>").text(""+this.options.closeText).html()}),"draggable"===d&&(c=f.is(":data(ui-draggable)"),c&&!a&&f.draggable("destroy"),!c&&a&&this._makeDraggable()),"position"===d&&this._position(),"resizable"===d&&(g=f.is(":data(ui-resizable)"),g&&!a&&f.resizable("destroy"),g&&"string"==typeof a&&f.resizable("option","handles",a),g||a===!1||this._makeResizable()),"title"===d&&this._title(this.uiDialogTitlebar.find(".ui-dialog-title")))
},_size:function(){var c,f,a,d=this.options;
this.element.show().css({width:"auto",minHeight:0,maxHeight:"none",height:0}),d.minWidth>d.width&&(d.width=d.minWidth),c=this.uiDialog.css({height:"auto",width:d.width}).outerHeight(),f=Math.max(0,d.minHeight-c),a="number"==typeof d.maxHeight?Math.max(0,d.maxHeight-c):"none","auto"===d.height?this.element.css({minHeight:f,maxHeight:a,height:"auto"}):this.element.height(Math.max(0,d.height-c)),this.uiDialog.is(":data(ui-resizable)")&&this.uiDialog.resizable("option","minHeight",this._minHeight())
},_blockFrames:function(){this.iframeBlocks=this.document.find("iframe").map(function(){var a=k(this);
return k("<div>").css({position:"absolute",width:a.outerWidth(),height:a.outerHeight()}).appendTo(a.parent()).offset(a.offset())[0]
})
},_unblockFrames:function(){this.iframeBlocks&&(this.iframeBlocks.remove(),delete this.iframeBlocks)
},_allowInteraction:function(a){return k(a.target).closest(".ui-dialog").length?!0:!!k(a.target).closest(".ui-datepicker").length
},_createOverlay:function(){if(this.options.modal){var a=!0;
this._delay(function(){a=!1
}),this.document.data("ui-dialog-overlays")||this._on(this.document,{focusin:function(c){a||this._allowInteraction(c)||(c.preventDefault(),this._trackingInstances()[0]._focusTabbable())
}}),this.overlay=k("<div>").appendTo(this._appendTo()),this._addClass(this.overlay,null,"ui-widget-overlay ui-front"),this._on(this.overlay,{mousedown:"_keepFocus"}),this.document.data("ui-dialog-overlays",(this.document.data("ui-dialog-overlays")||0)+1)
}},_destroyOverlay:function(){if(this.options.modal&&this.overlay){var a=this.document.data("ui-dialog-overlays")-1;
a?this.document.data("ui-dialog-overlays",a):(this._off(this.document,"focusin"),this.document.removeData("ui-dialog-overlays")),this.overlay.remove(),this.overlay=null
}}}),k.uiBackCompat!==!1&&k.widget("ui.dialog",k.ui.dialog,{options:{dialogClass:""},_createWrapper:function(){this._super(),this.uiDialog.addClass(this.options.dialogClass)
},_setOption:function(a,c){"dialogClass"===a&&this.uiDialog.removeClass(this.options.dialogClass).addClass(c),this._superApply(arguments)
}}),k.ui.dialog,k.widget("ui.droppable",{version:"1.12.1",widgetEventPrefix:"drop",options:{accept:"*",addClasses:!0,greedy:!1,scope:"default",tolerance:"intersect",activate:null,deactivate:null,drop:null,out:null,over:null},_create:function(){var d,a=this.options,c=a.accept;
this.isover=!1,this.isout=!0,this.accept=k.isFunction(c)?c:function(e){return e.is(c)
},this.proportions=function(){return arguments.length?(d=arguments[0],void 0):d?d:d={width:this.element[0].offsetWidth,height:this.element[0].offsetHeight}
},this._addToManager(a.scope),a.addClasses&&this._addClass("ui-droppable")
},_addToManager:function(a){k.ui.ddmanager.droppables[a]=k.ui.ddmanager.droppables[a]||[],k.ui.ddmanager.droppables[a].push(this)
},_splice:function(a){for(var c=0;
a.length>c;
c++){a[c]===this&&a.splice(c,1)
}},_destroy:function(){var a=k.ui.ddmanager.droppables[this.options.scope];
this._splice(a)
},_setOption:function(d,a){if("accept"===d){this.accept=k.isFunction(a)?a:function(e){return e.is(a)
}
}else{if("scope"===d){var c=k.ui.ddmanager.droppables[this.options.scope];
this._splice(c),this._addToManager(a)
}}this._super(d,a)
},_activate:function(c){var a=k.ui.ddmanager.current;
this._addActiveClass(),a&&this._trigger("activate",c,this.ui(a))
},_deactivate:function(c){var a=k.ui.ddmanager.current;
this._removeActiveClass(),a&&this._trigger("deactivate",c,this.ui(a))
},_over:function(c){var a=k.ui.ddmanager.current;
a&&(a.currentItem||a.element)[0]!==this.element[0]&&this.accept.call(this.element[0],a.currentItem||a.element)&&(this._addHoverClass(),this._trigger("over",c,this.ui(a)))
},_out:function(c){var a=k.ui.ddmanager.current;
a&&(a.currentItem||a.element)[0]!==this.element[0]&&this.accept.call(this.element[0],a.currentItem||a.element)&&(this._removeHoverClass(),this._trigger("out",c,this.ui(a)))
},_drop:function(d,a){var c=a||k.ui.ddmanager.current,f=!1;
return c&&(c.currentItem||c.element)[0]!==this.element[0]?(this.element.find(":data(ui-droppable)").not(".ui-draggable-dragging").each(function(){var e=k(this).droppable("instance");
return e.options.greedy&&!e.options.disabled&&e.options.scope===c.options.scope&&e.accept.call(e.element[0],c.currentItem||c.element)&&b(c,k.extend(e,{offset:e.element.offset()}),e.options.tolerance,d)?(f=!0,!1):void 0
}),f?!1:this.accept.call(this.element[0],c.currentItem||c.element)?(this._removeActiveClass(),this._removeHoverClass(),this._trigger("drop",d,this.ui(c)),this.element):!1):!1
},ui:function(a){return{draggable:a.currentItem||a.element,helper:a.helper,position:a.position,offset:a.positionAbs}
},_addHoverClass:function(){this._addClass("ui-droppable-hover")
},_removeHoverClass:function(){this._removeClass("ui-droppable-hover")
},_addActiveClass:function(){this._addClass("ui-droppable-active")
},_removeActiveClass:function(){this._removeClass("ui-droppable-active")
}});
var b=k.ui.intersect=function(){function a(d,f,c){return d>=f&&f+c>d
}return function(L,t,Q,m){if(!t.offset){return !1
}var g=(L.positionAbs||L.position.absolute).left+L.margins.left,O=(L.positionAbs||L.position.absolute).top+L.margins.top,f=g+L.helperProportions.width,v=O+L.helperProportions.height,p=t.offset.left,N=t.offset.top,P=p+t.proportions().width,M=N+t.proportions().height;
switch(Q){case"fit":return g>=p&&P>=f&&O>=N&&M>=v;
case"intersect":return g+L.helperProportions.width/2>p&&P>f-L.helperProportions.width/2&&O+L.helperProportions.height/2>N&&M>v-L.helperProportions.height/2;
case"pointer":return a(m.pageY,N,t.proportions().height)&&a(m.pageX,p,t.proportions().width);
case"touch":return(O>=N&&M>=O||v>=N&&M>=v||N>O&&v>M)&&(g>=p&&P>=g||f>=p&&P>=f||p>g&&f>P);
default:return !1
}}
}();
k.ui.ddmanager={current:null,droppables:{"default":[]},prepareOffsets:function(h,d){var f,m,l=k.ui.ddmanager.droppables[h.options.scope]||[],c=d?d.type:null,g=(h.currentItem||h.element).find(":data(ui-droppable)").addBack();
k:for(f=0;
l.length>f;
f++){if(!(l[f].options.disabled||h&&!l[f].accept.call(l[f].element[0],h.currentItem||h.element))){for(m=0;
g.length>m;
m++){if(g[m]===l[f].element[0]){l[f].proportions().height=0;
continue k
}}l[f].visible="none"!==l[f].element.css("display"),l[f].visible&&("mousedown"===c&&l[f]._activate.call(l[f],d),l[f].offset=l[f].element.offset(),l[f].proportions({width:l[f].element[0].offsetWidth,height:l[f].element[0].offsetHeight}))
}}},drop:function(d,a){var c=!1;
return k.each((k.ui.ddmanager.droppables[d.options.scope]||[]).slice(),function(){this.options&&(!this.options.disabled&&this.visible&&b(d,this,this.options.tolerance,a)&&(c=this._drop.call(this,a)||c),!this.options.disabled&&this.visible&&this.accept.call(this.element[0],d.currentItem||d.element)&&(this.isout=!0,this.isover=!1,this._deactivate.call(this,a)))
}),c
},dragStart:function(c,a){c.element.parentsUntil("body").on("scroll.droppable",function(){c.options.refreshPositions||k.ui.ddmanager.prepareOffsets(c,a)
})
},drag:function(c,a){c.options.refreshPositions&&k.ui.ddmanager.prepareOffsets(c,a),k.each(k.ui.ddmanager.droppables[c.options.scope]||[],function(){if(!this.options.disabled&&!this.greedyChild&&this.visible){var e,h,g,d=b(c,this,this.options.tolerance,a),f=!d&&this.isover?"isout":d&&!this.isover?"isover":null;
f&&(this.options.greedy&&(h=this.options.scope,g=this.element.parents(":data(ui-droppable)").filter(function(){return k(this).droppable("instance").options.scope===h
}),g.length&&(e=k(g[0]).droppable("instance"),e.greedyChild="isover"===f)),e&&"isover"===f&&(e.isover=!1,e.isout=!0,e._out.call(e,a)),this[f]=!0,this["isout"===f?"isover":"isout"]=!1,this["isover"===f?"_over":"_out"].call(this,a),e&&"isout"===f&&(e.isout=!1,e.isover=!0,e._over.call(e,a)))
}})
},dragStop:function(c,a){c.element.parentsUntil("body").off("scroll.droppable"),c.options.refreshPositions||k.ui.ddmanager.prepareOffsets(c,a)
}},k.uiBackCompat!==!1&&k.widget("ui.droppable",k.ui.droppable,{options:{hoverClass:!1,activeClass:!1},_addActiveClass:function(){this._super(),this.options.activeClass&&this.element.addClass(this.options.activeClass)
},_removeActiveClass:function(){this._super(),this.options.activeClass&&this.element.removeClass(this.options.activeClass)
},_addHoverClass:function(){this._super(),this.options.hoverClass&&this.element.addClass(this.options.hoverClass)
},_removeHoverClass:function(){this._super(),this.options.hoverClass&&this.element.removeClass(this.options.hoverClass)
}}),k.ui.droppable,k.widget("ui.progressbar",{version:"1.12.1",options:{classes:{"ui-progressbar":"ui-corner-all","ui-progressbar-value":"ui-corner-left","ui-progressbar-complete":"ui-corner-right"},max:100,value:0,change:null,complete:null},min:0,_create:function(){this.oldValue=this.options.value=this._constrainedValue(),this.element.attr({role:"progressbar","aria-valuemin":this.min}),this._addClass("ui-progressbar","ui-widget ui-widget-content"),this.valueDiv=k("<div>").appendTo(this.element),this._addClass(this.valueDiv,"ui-progressbar-value","ui-widget-header"),this._refreshValue()
},_destroy:function(){this.element.removeAttr("role aria-valuemin aria-valuemax aria-valuenow"),this.valueDiv.remove()
},value:function(a){return void 0===a?this.options.value:(this.options.value=this._constrainedValue(a),this._refreshValue(),void 0)
},_constrainedValue:function(a){return void 0===a&&(a=this.options.value),this.indeterminate=a===!1,"number"!=typeof a&&(a=0),this.indeterminate?!1:Math.min(this.options.max,Math.max(this.min,a))
},_setOptions:function(a){var c=a.value;
delete a.value,this._super(a),this.options.value=this._constrainedValue(c),this._refreshValue()
},_setOption:function(a,c){"max"===a&&(c=Math.max(this.min,c)),this._super(a,c)
},_setOptionDisabled:function(a){this._super(a),this.element.attr("aria-disabled",a),this._toggleClass(null,"ui-state-disabled",!!a)
},_percentage:function(){return this.indeterminate?100:100*(this.options.value-this.min)/(this.options.max-this.min)
},_refreshValue:function(){var c=this.options.value,a=this._percentage();
this.valueDiv.toggle(this.indeterminate||c>this.min).width(a.toFixed(0)+"%"),this._toggleClass(this.valueDiv,"ui-progressbar-complete",null,c===this.options.max)._toggleClass("ui-progressbar-indeterminate",null,this.indeterminate),this.indeterminate?(this.element.removeAttr("aria-valuenow"),this.overlayDiv||(this.overlayDiv=k("<div>").appendTo(this.valueDiv),this._addClass(this.overlayDiv,"ui-progressbar-overlay"))):(this.element.attr({"aria-valuemax":this.options.max,"aria-valuenow":c}),this.overlayDiv&&(this.overlayDiv.remove(),this.overlayDiv=null)),this.oldValue!==c&&(this.oldValue=c,this._trigger("change")),c===this.options.max&&this._trigger("complete")
}}),k.widget("ui.selectable",k.ui.mouse,{version:"1.12.1",options:{appendTo:"body",autoRefresh:!0,distance:0,filter:"*",tolerance:"touch",selected:null,selecting:null,start:null,stop:null,unselected:null,unselecting:null},_create:function(){var a=this;
this._addClass("ui-selectable"),this.dragged=!1,this.refresh=function(){a.elementPos=k(a.element[0]).offset(),a.selectees=k(a.options.filter,a.element[0]),a._addClass(a.selectees,"ui-selectee"),a.selectees.each(function(){var c=k(this),d=c.offset(),e={left:d.left-a.elementPos.left,top:d.top-a.elementPos.top};
k.data(this,"selectable-item",{element:this,$element:c,left:e.left,top:e.top,right:e.left+c.outerWidth(),bottom:e.top+c.outerHeight(),startselected:!1,selected:c.hasClass("ui-selected"),selecting:c.hasClass("ui-selecting"),unselecting:c.hasClass("ui-unselecting")})
})
},this.refresh(),this._mouseInit(),this.helper=k("<div>"),this._addClass(this.helper,"ui-selectable-helper")
},_destroy:function(){this.selectees.removeData("selectable-item"),this._mouseDestroy()
},_mouseStart:function(d){var a=this,c=this.options;
this.opos=[d.pageX,d.pageY],this.elementPos=k(this.element[0]).offset(),this.options.disabled||(this.selectees=k(c.filter,this.element[0]),this._trigger("start",d),k(c.appendTo).append(this.helper),this.helper.css({left:d.pageX,top:d.pageY,width:0,height:0}),c.autoRefresh&&this.refresh(),this.selectees.filter(".ui-selected").each(function(){var e=k.data(this,"selectable-item");
e.startselected=!0,d.metaKey||d.ctrlKey||(a._removeClass(e.$element,"ui-selected"),e.selected=!1,a._addClass(e.$element,"ui-unselecting"),e.unselecting=!0,a._trigger("unselecting",d,{unselecting:e.element}))
}),k(d.target).parents().addBack().each(function(){var e,f=k.data(this,"selectable-item");
return f?(e=!d.metaKey&&!d.ctrlKey||!f.$element.hasClass("ui-selected"),a._removeClass(f.$element,e?"ui-unselecting":"ui-selected")._addClass(f.$element,e?"ui-selecting":"ui-unselecting"),f.unselecting=!e,f.selecting=e,f.selected=e,e?a._trigger("selecting",d,{selecting:f.element}):a._trigger("unselecting",d,{unselecting:f.element}),!1):void 0
}))
},_mouseDrag:function(m){if(this.dragged=!0,!this.options.disabled){var d,g=this,t=this.options,p=this.opos[0],c=this.opos[1],l=m.pageX,f=m.pageY;
return p>l&&(d=l,l=p,p=d),c>f&&(d=f,f=c,c=d),this.helper.css({left:p,top:c,width:l-p,height:f-c}),this.selectees.each(function(){var e=k.data(this,"selectable-item"),a=!1,h={};
e&&e.element!==g.element[0]&&(h.left=e.left+g.elementPos.left,h.right=e.right+g.elementPos.left,h.top=e.top+g.elementPos.top,h.bottom=e.bottom+g.elementPos.top,"touch"===t.tolerance?a=!(h.left>l||p>h.right||h.top>f||c>h.bottom):"fit"===t.tolerance&&(a=h.left>p&&l>h.right&&h.top>c&&f>h.bottom),a?(e.selected&&(g._removeClass(e.$element,"ui-selected"),e.selected=!1),e.unselecting&&(g._removeClass(e.$element,"ui-unselecting"),e.unselecting=!1),e.selecting||(g._addClass(e.$element,"ui-selecting"),e.selecting=!0,g._trigger("selecting",m,{selecting:e.element}))):(e.selecting&&((m.metaKey||m.ctrlKey)&&e.startselected?(g._removeClass(e.$element,"ui-selecting"),e.selecting=!1,g._addClass(e.$element,"ui-selected"),e.selected=!0):(g._removeClass(e.$element,"ui-selecting"),e.selecting=!1,e.startselected&&(g._addClass(e.$element,"ui-unselecting"),e.unselecting=!0),g._trigger("unselecting",m,{unselecting:e.element}))),e.selected&&(m.metaKey||m.ctrlKey||e.startselected||(g._removeClass(e.$element,"ui-selected"),e.selected=!1,g._addClass(e.$element,"ui-unselecting"),e.unselecting=!0,g._trigger("unselecting",m,{unselecting:e.element})))))
}),!1
}},_mouseStop:function(c){var a=this;
return this.dragged=!1,k(".ui-unselecting",this.element[0]).each(function(){var d=k.data(this,"selectable-item");
a._removeClass(d.$element,"ui-unselecting"),d.unselecting=!1,d.startselected=!1,a._trigger("unselected",c,{unselected:d.element})
}),k(".ui-selecting",this.element[0]).each(function(){var d=k.data(this,"selectable-item");
a._removeClass(d.$element,"ui-selecting")._addClass(d.$element,"ui-selected"),d.selecting=!1,d.selected=!0,d.startselected=!0,a._trigger("selected",c,{selected:d.element})
}),this._trigger("stop",c),this.helper.remove(),!1
}}),k.widget("ui.selectmenu",[k.ui.formResetMixin,{version:"1.12.1",defaultElement:"<select>",options:{appendTo:null,classes:{"ui-selectmenu-button-open":"ui-corner-top","ui-selectmenu-button-closed":"ui-corner-all"},disabled:null,icons:{button:"ui-icon-triangle-1-s"},position:{my:"left top",at:"left bottom",collision:"none"},width:!1,change:null,close:null,focus:null,open:null,select:null},_create:function(){var a=this.element.uniqueId().attr("id");
this.ids={element:a,button:a+"-button",menu:a+"-menu"},this._drawButton(),this._drawMenu(),this._bindFormResetHandler(),this._rendered=!1,this.menuItems=k()
},_drawButton:function(){var d,a=this,c=this._parseOption(this.element.find("option:selected"),this.element[0].selectedIndex);
this.labels=this.element.labels().attr("for",this.ids.button),this._on(this.labels,{click:function(e){this.button.focus(),e.preventDefault()
}}),this.element.hide(),this.button=k("<span>",{tabindex:this.options.disabled?-1:0,id:this.ids.button,role:"combobox","aria-expanded":"false","aria-autocomplete":"list","aria-owns":this.ids.menu,"aria-haspopup":"true",title:this.element.attr("title")}).insertAfter(this.element),this._addClass(this.button,"ui-selectmenu-button ui-selectmenu-button-closed","ui-button ui-widget"),d=k("<span>").appendTo(this.button),this._addClass(d,"ui-selectmenu-icon","ui-icon "+this.options.icons.button),this.buttonItem=this._renderButtonItem(c).appendTo(this.button),this.options.width!==!1&&this._resizeButton(),this._on(this.button,this._buttonEvents),this.button.one("focusin",function(){a._rendered||a._refreshMenu()
})
},_drawMenu:function(){var a=this;
this.menu=k("<ul>",{"aria-hidden":"true","aria-labelledby":this.ids.button,id:this.ids.menu}),this.menuWrap=k("<div>").append(this.menu),this._addClass(this.menuWrap,"ui-selectmenu-menu","ui-front"),this.menuWrap.appendTo(this._appendTo()),this.menuInstance=this.menu.menu({classes:{"ui-menu":"ui-corner-bottom"},role:"listbox",select:function(d,c){d.preventDefault(),a._setSelection(),a._select(c.item.data("ui-selectmenu-item"),d)
},focus:function(d,c){var e=c.item.data("ui-selectmenu-item");
null!=a.focusIndex&&e.index!==a.focusIndex&&(a._trigger("focus",d,{item:e}),a.isOpen||a._select(e,d)),a.focusIndex=e.index,a.button.attr("aria-activedescendant",a.menuItems.eq(e.index).attr("id"))
}}).menu("instance"),this.menuInstance._off(this.menu,"mouseleave"),this.menuInstance._closeOnDocumentClick=function(){return !1
},this.menuInstance._isDivider=function(){return !1
}
},refresh:function(){this._refreshMenu(),this.buttonItem.replaceWith(this.buttonItem=this._renderButtonItem(this._getSelectedItem().data("ui-selectmenu-item")||{})),null===this.options.width&&this._resizeButton()
},_refreshMenu:function(){var a,c=this.element.find("option");
this.menu.empty(),this._parseOptions(c),this._renderMenu(this.menu,this.items),this.menuInstance.refresh(),this.menuItems=this.menu.find("li").not(".ui-selectmenu-optgroup").find(".ui-menu-item-wrapper"),this._rendered=!0,c.length&&(a=this._getSelectedItem(),this.menuInstance.focus(null,a),this._setAria(a.data("ui-selectmenu-item")),this._setOption("disabled",this.element.prop("disabled")))
},open:function(a){this.options.disabled||(this._rendered?(this._removeClass(this.menu.find(".ui-state-active"),null,"ui-state-active"),this.menuInstance.focus(null,this._getSelectedItem())):this._refreshMenu(),this.menuItems.length&&(this.isOpen=!0,this._toggleAttr(),this._resizeMenu(),this._position(),this._on(this.document,this._documentClick),this._trigger("open",a)))
},_position:function(){this.menuWrap.position(k.extend({of:this.button},this.options.position))
},close:function(a){this.isOpen&&(this.isOpen=!1,this._toggleAttr(),this.range=null,this._off(this.document),this._trigger("close",a))
},widget:function(){return this.button
},menuWidget:function(){return this.menu
},_renderButtonItem:function(c){var a=k("<span>");
return this._setText(a,c.label),this._addClass(a,"ui-selectmenu-text"),a
},_renderMenu:function(d,a){var c=this,f="";
k.each(a,function(g,h){var e;
h.optgroup!==f&&(e=k("<li>",{text:h.optgroup}),c._addClass(e,"ui-selectmenu-optgroup","ui-menu-divider"+(h.element.parent("optgroup").prop("disabled")?" ui-state-disabled":"")),e.appendTo(d),f=h.optgroup),c._renderItemData(d,h)
})
},_renderItemData:function(a,c){return this._renderItem(a,c).data("ui-selectmenu-item",c)
},_renderItem:function(d,a){var c=k("<li>"),f=k("<div>",{title:a.element.attr("title")});
return a.disabled&&this._addClass(c,null,"ui-state-disabled"),this._setText(f,a.label),c.append(f).appendTo(d)
},_setText:function(a,c){c?a.text(c):a.html("&#160;")
},_move:function(c,f){var a,d,g=".ui-menu-item";
this.isOpen?a=this.menuItems.eq(this.focusIndex).parent("li"):(a=this.menuItems.eq(this.element[0].selectedIndex).parent("li"),g+=":not(.ui-state-disabled)"),d="first"===c||"last"===c?a["first"===c?"prevAll":"nextAll"](g).eq(-1):a[c+"All"](g).eq(0),d.length&&this.menuInstance.focus(f,d)
},_getSelectedItem:function(){return this.menuItems.eq(this.element[0].selectedIndex).parent("li")
},_toggle:function(a){this[this.isOpen?"close":"open"](a)
},_setSelection:function(){var a;
this.range&&(window.getSelection?(a=window.getSelection(),a.removeAllRanges(),a.addRange(this.range)):this.range.select(),this.button.focus())
},_documentClick:{mousedown:function(a){this.isOpen&&(k(a.target).closest(".ui-selectmenu-menu, #"+k.ui.escapeSelector(this.ids.button)).length||this.close(a))
}},_buttonEvents:{mousedown:function(){var a;
window.getSelection?(a=window.getSelection(),a.rangeCount&&(this.range=a.getRangeAt(0))):this.range=document.selection.createRange()
},click:function(a){this._setSelection(),this._toggle(a)
},keydown:function(c){var a=!0;
switch(c.keyCode){case k.ui.keyCode.TAB:case k.ui.keyCode.ESCAPE:this.close(c),a=!1;
break;
case k.ui.keyCode.ENTER:this.isOpen&&this._selectFocusedItem(c);
break;
case k.ui.keyCode.UP:c.altKey?this._toggle(c):this._move("prev",c);
break;
case k.ui.keyCode.DOWN:c.altKey?this._toggle(c):this._move("next",c);
break;
case k.ui.keyCode.SPACE:this.isOpen?this._selectFocusedItem(c):this._toggle(c);
break;
case k.ui.keyCode.LEFT:this._move("prev",c);
break;
case k.ui.keyCode.RIGHT:this._move("next",c);
break;
case k.ui.keyCode.HOME:case k.ui.keyCode.PAGE_UP:this._move("first",c);
break;
case k.ui.keyCode.END:case k.ui.keyCode.PAGE_DOWN:this._move("last",c);
break;
default:this.menu.trigger(c),a=!1
}a&&c.preventDefault()
}},_selectFocusedItem:function(a){var c=this.menuItems.eq(this.focusIndex).parent("li");
c.hasClass("ui-state-disabled")||this._select(c.data("ui-selectmenu-item"),a)
},_select:function(c,d){var a=this.element[0].selectedIndex;
this.element[0].selectedIndex=c.index,this.buttonItem.replaceWith(this.buttonItem=this._renderButtonItem(c)),this._setAria(c),this._trigger("select",d,{item:c}),c.index!==a&&this._trigger("change",d,{item:c}),this.close(d)
},_setAria:function(a){var c=this.menuItems.eq(a.index).attr("id");
this.button.attr({"aria-labelledby":c,"aria-activedescendant":c}),this.menu.attr("aria-activedescendant",c)
},_setOption:function(c,d){if("icons"===c){var a=this.button.find("span.ui-icon");
this._removeClass(a,null,this.options.icons.button)._addClass(a,null,d.button)
}this._super(c,d),"appendTo"===c&&this.menuWrap.appendTo(this._appendTo()),"width"===c&&this._resizeButton()
},_setOptionDisabled:function(a){this._super(a),this.menuInstance.option("disabled",a),this.button.attr("aria-disabled",a),this._toggleClass(this.button,null,"ui-state-disabled",a),this.element.prop("disabled",a),a?(this.button.attr("tabindex",-1),this.close()):this.button.attr("tabindex",0)
},_appendTo:function(){var a=this.options.appendTo;
return a&&(a=a.jquery||a.nodeType?k(a):this.document.find(a).eq(0)),a&&a[0]||(a=this.element.closest(".ui-front, dialog")),a.length||(a=this.document[0].body),a
},_toggleAttr:function(){this.button.attr("aria-expanded",this.isOpen),this._removeClass(this.button,"ui-selectmenu-button-"+(this.isOpen?"closed":"open"))._addClass(this.button,"ui-selectmenu-button-"+(this.isOpen?"open":"closed"))._toggleClass(this.menuWrap,"ui-selectmenu-open",null,this.isOpen),this.menu.attr("aria-hidden",!this.isOpen)
},_resizeButton:function(){var a=this.options.width;
return a===!1?(this.button.css("width",""),void 0):(null===a&&(a=this.element.show().outerWidth(),this.element.hide()),this.button.outerWidth(a),void 0)
},_resizeMenu:function(){this.menu.outerWidth(Math.max(this.button.outerWidth(),this.menu.width("").outerWidth()+1))
},_getCreateOptions:function(){var a=this._super();
return a.disabled=this.element.prop("disabled"),a
},_parseOptions:function(d){var a=this,c=[];
d.each(function(f,g){c.push(a._parseOption(k(g),f))
}),this.items=c
},_parseOption:function(c,d){var a=c.parent("optgroup");
return{element:c,index:d,value:c.val(),label:c.text(),optgroup:a.attr("label")||"",disabled:a.prop("disabled")||c.prop("disabled")}
},_destroy:function(){this._unbindFormResetHandler(),this.menuWrap.remove(),this.button.remove(),this.element.show(),this.element.removeUniqueId(),this.labels.attr("for",this.ids.element)
}}]),k.widget("ui.slider",k.ui.mouse,{version:"1.12.1",widgetEventPrefix:"slide",options:{animate:!1,classes:{"ui-slider":"ui-corner-all","ui-slider-handle":"ui-corner-all","ui-slider-range":"ui-corner-all ui-widget-header"},distance:0,max:100,min:0,orientation:"horizontal",range:!1,step:1,value:0,values:null,change:null,slide:null,start:null,stop:null},numPages:5,_create:function(){this._keySliding=!1,this._mouseSliding=!1,this._animateOff=!0,this._handleIndex=null,this._detectOrientation(),this._mouseInit(),this._calculateNewMax(),this._addClass("ui-slider ui-slider-"+this.orientation,"ui-widget ui-widget-content"),this._refresh(),this._animateOff=!1
},_refresh:function(){this._createRange(),this._createHandles(),this._setupEvents(),this._refreshValue()
},_createHandles:function(){var g,d,f=this.options,l=this.element.find(".ui-slider-handle"),h="<span tabindex='0'></span>",c=[];
for(d=f.values&&f.values.length||1,l.length>d&&(l.slice(d).remove(),l=l.slice(0,d)),g=l.length;
d>g;
g++){c.push(h)
}this.handles=l.add(k(c.join("")).appendTo(this.element)),this._addClass(this.handles,"ui-slider-handle","ui-state-default"),this.handle=this.handles.eq(0),this.handles.each(function(a){k(this).data("ui-slider-handle-index",a).attr("tabIndex",0)
})
},_createRange:function(){var a=this.options;
a.range?(a.range===!0&&(a.values?a.values.length&&2!==a.values.length?a.values=[a.values[0],a.values[0]]:k.isArray(a.values)&&(a.values=a.values.slice(0)):a.values=[this._valueMin(),this._valueMin()]),this.range&&this.range.length?(this._removeClass(this.range,"ui-slider-range-min ui-slider-range-max"),this.range.css({left:"",bottom:""})):(this.range=k("<div>").appendTo(this.element),this._addClass(this.range,"ui-slider-range")),("min"===a.range||"max"===a.range)&&this._addClass(this.range,"ui-slider-range-"+a.range)):(this.range&&this.range.remove(),this.range=null)
},_setupEvents:function(){this._off(this.handles),this._on(this.handles,this._handleEvents),this._hoverable(this.handles),this._focusable(this.handles)
},_destroy:function(){this.handles.remove(),this.range&&this.range.remove(),this._mouseDestroy()
},_mouseCapture:function(v){var p,O,g,f,M,d,t,m,L=this,N=this.options;
return N.disabled?!1:(this.elementSize={width:this.element.outerWidth(),height:this.element.outerHeight()},this.elementOffset=this.element.offset(),p={x:v.pageX,y:v.pageY},O=this._normValueFromMouse(p),g=this._valueMax()-this._valueMin()+1,this.handles.each(function(c){var a=Math.abs(O-L.values(c));
(g>a||g===a&&(c===L._lastChangedValue||L.values(c)===N.min))&&(g=a,f=k(this),M=c)
}),d=this._start(v,M),d===!1?!1:(this._mouseSliding=!0,this._handleIndex=M,this._addClass(f,null,"ui-state-active"),f.trigger("focus"),t=f.offset(),m=!k(v.target).parents().addBack().is(".ui-slider-handle"),this._clickOffset=m?{left:0,top:0}:{left:v.pageX-t.left-f.width()/2,top:v.pageY-t.top-f.height()/2-(parseInt(f.css("borderTopWidth"),10)||0)-(parseInt(f.css("borderBottomWidth"),10)||0)+(parseInt(f.css("marginTop"),10)||0)},this.handles.hasClass("ui-state-hover")||this._slide(v,M,O),this._animateOff=!0,!0))
},_mouseStart:function(){return !0
},_mouseDrag:function(c){var d={x:c.pageX,y:c.pageY},a=this._normValueFromMouse(d);
return this._slide(c,this._handleIndex,a),!1
},_mouseStop:function(a){return this._removeClass(this.handles,null,"ui-state-active"),this._mouseSliding=!1,this._stop(a,this._handleIndex),this._change(a,this._handleIndex),this._handleIndex=null,this._clickOffset=null,this._animateOff=!1,!1
},_detectOrientation:function(){this.orientation="vertical"===this.options.orientation?"vertical":"horizontal"
},_normValueFromMouse:function(c){var f,a,d,h,g;
return"horizontal"===this.orientation?(f=this.elementSize.width,a=c.x-this.elementOffset.left-(this._clickOffset?this._clickOffset.left:0)):(f=this.elementSize.height,a=c.y-this.elementOffset.top-(this._clickOffset?this._clickOffset.top:0)),d=a/f,d>1&&(d=1),0>d&&(d=0),"vertical"===this.orientation&&(d=1-d),h=this._valueMax()-this._valueMin(),g=this._valueMin()+d*h,this._trimAlignValue(g)
},_uiHash:function(c,f,a){var d={handle:this.handles[c],handleIndex:c,value:void 0!==f?f:this.value()};
return this._hasMultipleValues()&&(d.value=void 0!==f?f:this.values(c),d.values=a||this.values()),d
},_hasMultipleValues:function(){return this.options.values&&this.options.values.length
},_start:function(a,c){return this._trigger("start",a,this._uiHash(c))
},_slide:function(f,h,d){var g,m,l=this.value(),c=this.values();
this._hasMultipleValues()&&(m=this.values(h?0:1),l=this.values(h),2===this.options.values.length&&this.options.range===!0&&(d=0===h?Math.min(m,d):Math.max(m,d)),c[h]=d),d!==l&&(g=this._trigger("slide",f,this._uiHash(h,d,c)),g!==!1&&(this._hasMultipleValues()?this.values(h,d):this.value(d)))
},_stop:function(a,c){this._trigger("stop",a,this._uiHash(c))
},_change:function(a,c){this._keySliding||this._mouseSliding||(this._lastChangedValue=c,this._trigger("change",a,this._uiHash(c)))
},value:function(a){return arguments.length?(this.options.value=this._trimAlignValue(a),this._refreshValue(),this._change(null,0),void 0):this._value()
},values:function(d,a){var c,g,f;
if(arguments.length>1){return this.options.values[d]=this._trimAlignValue(a),this._refreshValue(),this._change(null,d),void 0
}if(!arguments.length){return this._values()
}if(!k.isArray(arguments[0])){return this._hasMultipleValues()?this._values(d):this.value()
}for(c=this.options.values,g=arguments[0],f=0;
c.length>f;
f+=1){c[f]=this._trimAlignValue(g[f]),this._change(null,f)
}this._refreshValue()
},_setOption:function(d,a){var c,f=0;
switch("range"===d&&this.options.range===!0&&("min"===a?(this.options.value=this._values(0),this.options.values=null):"max"===a&&(this.options.value=this._values(this.options.values.length-1),this.options.values=null)),k.isArray(this.options.values)&&(f=this.options.values.length),this._super(d,a),d){case"orientation":this._detectOrientation(),this._removeClass("ui-slider-horizontal ui-slider-vertical")._addClass("ui-slider-"+this.orientation),this._refreshValue(),this.options.range&&this._refreshRange(a),this.handles.css("horizontal"===a?"bottom":"left","");
break;
case"value":this._animateOff=!0,this._refreshValue(),this._change(null,0),this._animateOff=!1;
break;
case"values":for(this._animateOff=!0,this._refreshValue(),c=f-1;
c>=0;
c--){this._change(null,c)
}this._animateOff=!1;
break;
case"step":case"min":case"max":this._animateOff=!0,this._calculateNewMax(),this._refreshValue(),this._animateOff=!1;
break;
case"range":this._animateOff=!0,this._refresh(),this._animateOff=!1
}},_setOptionDisabled:function(a){this._super(a),this._toggleClass(null,"ui-state-disabled",!!a)
},_value:function(){var a=this.options.value;
return a=this._trimAlignValue(a)
},_values:function(c){var f,a,d;
if(arguments.length){return f=this.options.values[c],f=this._trimAlignValue(f)
}if(this._hasMultipleValues()){for(a=this.options.values.slice(),d=0;
a.length>d;
d+=1){a[d]=this._trimAlignValue(a[d])
}return a
}return[]
},_trimAlignValue:function(c){if(this._valueMin()>=c){return this._valueMin()
}if(c>=this._valueMax()){return this._valueMax()
}var f=this.options.step>0?this.options.step:1,a=(c-this._valueMin())%f,d=c-a;
return 2*Math.abs(a)>=f&&(d+=a>0?f:-f),parseFloat(d.toFixed(5))
},_calculateNewMax:function(){var c=this.options.max,f=this._valueMin(),a=this.options.step,d=Math.round((c-f)/a)*a;
c=d+f,c>this.options.max&&(c-=a),this.max=parseFloat(c.toFixed(this._precision()))
},_precision:function(){var a=this._precisionOf(this.options.step);
return null!==this.options.min&&(a=Math.max(a,this._precisionOf(this.options.min))),a
},_precisionOf:function(c){var d=""+c,a=d.indexOf(".");
return -1===a?0:d.length-a-1
},_valueMin:function(){return this.options.min
},_valueMax:function(){return this.max
},_refreshRange:function(a){"vertical"===a&&this.range.css({width:"",left:""}),"horizontal"===a&&this.range.css({height:"",bottom:""})
},_refreshValue:function(){var u,p,M,g,f,L=this.options.range,d=this.options,t=this,m=this._animateOff?!1:d.animate,v={};
this._hasMultipleValues()?this.handles.each(function(a){p=100*((t.values(a)-t._valueMin())/(t._valueMax()-t._valueMin())),v["horizontal"===t.orientation?"left":"bottom"]=p+"%",k(this).stop(1,1)[m?"animate":"css"](v,d.animate),t.options.range===!0&&("horizontal"===t.orientation?(0===a&&t.range.stop(1,1)[m?"animate":"css"]({left:p+"%"},d.animate),1===a&&t.range[m?"animate":"css"]({width:p-u+"%"},{queue:!1,duration:d.animate})):(0===a&&t.range.stop(1,1)[m?"animate":"css"]({bottom:p+"%"},d.animate),1===a&&t.range[m?"animate":"css"]({height:p-u+"%"},{queue:!1,duration:d.animate}))),u=p
}):(M=this.value(),g=this._valueMin(),f=this._valueMax(),p=f!==g?100*((M-g)/(f-g)):0,v["horizontal"===this.orientation?"left":"bottom"]=p+"%",this.handle.stop(1,1)[m?"animate":"css"](v,d.animate),"min"===L&&"horizontal"===this.orientation&&this.range.stop(1,1)[m?"animate":"css"]({width:p+"%"},d.animate),"max"===L&&"horizontal"===this.orientation&&this.range.stop(1,1)[m?"animate":"css"]({width:100-p+"%"},d.animate),"min"===L&&"vertical"===this.orientation&&this.range.stop(1,1)[m?"animate":"css"]({height:p+"%"},d.animate),"max"===L&&"vertical"===this.orientation&&this.range.stop(1,1)[m?"animate":"css"]({height:100-p+"%"},d.animate))
},_handleEvents:{keydown:function(g){var d,f,l,h,c=k(g.target).data("ui-slider-handle-index");
switch(g.keyCode){case k.ui.keyCode.HOME:case k.ui.keyCode.END:case k.ui.keyCode.PAGE_UP:case k.ui.keyCode.PAGE_DOWN:case k.ui.keyCode.UP:case k.ui.keyCode.RIGHT:case k.ui.keyCode.DOWN:case k.ui.keyCode.LEFT:if(g.preventDefault(),!this._keySliding&&(this._keySliding=!0,this._addClass(k(g.target),null,"ui-state-active"),d=this._start(g,c),d===!1)){return
}}switch(h=this.options.step,f=l=this._hasMultipleValues()?this.values(c):this.value(),g.keyCode){case k.ui.keyCode.HOME:l=this._valueMin();
break;
case k.ui.keyCode.END:l=this._valueMax();
break;
case k.ui.keyCode.PAGE_UP:l=this._trimAlignValue(f+(this._valueMax()-this._valueMin())/this.numPages);
break;
case k.ui.keyCode.PAGE_DOWN:l=this._trimAlignValue(f-(this._valueMax()-this._valueMin())/this.numPages);
break;
case k.ui.keyCode.UP:case k.ui.keyCode.RIGHT:if(f===this._valueMax()){return
}l=this._trimAlignValue(f+h);
break;
case k.ui.keyCode.DOWN:case k.ui.keyCode.LEFT:if(f===this._valueMin()){return
}l=this._trimAlignValue(f-h)
}this._slide(g,c,l)
},keyup:function(c){var a=k(c.target).data("ui-slider-handle-index");
this._keySliding&&(this._keySliding=!1,this._stop(c,a),this._change(c,a),this._removeClass(k(c.target),null,"ui-state-active"))
}}}),k.widget("ui.sortable",k.ui.mouse,{version:"1.12.1",widgetEventPrefix:"sort",ready:!1,options:{appendTo:"parent",axis:!1,connectWith:!1,containment:!1,cursor:"auto",cursorAt:!1,dropOnEmpty:!0,forcePlaceholderSize:!1,forceHelperSize:!1,grid:!1,handle:!1,helper:"original",items:"> *",opacity:!1,placeholder:!1,revert:!1,scroll:!0,scrollSensitivity:20,scrollSpeed:20,scope:"default",tolerance:"intersect",zIndex:1000,activate:null,beforeStop:null,change:null,deactivate:null,out:null,over:null,receive:null,remove:null,sort:null,start:null,stop:null,update:null},_isOverAxis:function(c,d,a){return c>=d&&d+a>c
},_isFloating:function(a){return/left|right/.test(a.css("float"))||/inline|table-cell/.test(a.css("display"))
},_create:function(){this.containerCache={},this._addClass("ui-sortable"),this.refresh(),this.offset=this.element.offset(),this._mouseInit(),this._setHandleClassName(),this.ready=!0
},_setOption:function(a,c){this._super(a,c),"handle"===a&&this._setHandleClassName()
},_setHandleClassName:function(){var a=this;
this._removeClass(this.element.find(".ui-sortable-handle"),"ui-sortable-handle"),k.each(this.items,function(){a._addClass(this.instance.options.handle?this.item.find(this.instance.options.handle):this.item,"ui-sortable-handle")
})
},_destroy:function(){this._mouseDestroy();
for(var a=this.items.length-1;
a>=0;
a--){this.items[a].item.removeData(this.widgetName+"-item")
}return this
},_mouseCapture:function(d,a){var c=null,g=!1,f=this;
return this.reverting?!1:this.options.disabled||"static"===this.options.type?!1:(this._refreshItems(d),k(d.target).parents().each(function(){return k.data(this,f.widgetName+"-item")===f?(c=k(this),!1):void 0
}),k.data(d.target,f.widgetName+"-item")===f&&(c=k(d.target)),c?!this.options.handle||a||(k(this.options.handle,c).find("*").addBack().each(function(){this===d.target&&(g=!0)
}),g)?(this.currentItem=c,this._removeCurrentsFromItems(),!0):!1:!1)
},_mouseStart:function(g,d,f){var l,h,c=this.options;
if(this.currentContainer=this,this.refreshPositions(),this.helper=this._createHelper(g),this._cacheHelperProportions(),this._cacheMargins(),this.scrollParent=this.helper.scrollParent(),this.offset=this.currentItem.offset(),this.offset={top:this.offset.top-this.margins.top,left:this.offset.left-this.margins.left},k.extend(this.offset,{click:{left:g.pageX-this.offset.left,top:g.pageY-this.offset.top},parent:this._getParentOffset(),relative:this._getRelativeOffset()}),this.helper.css("position","absolute"),this.cssPosition=this.helper.css("position"),this.originalPosition=this._generatePosition(g),this.originalPageX=g.pageX,this.originalPageY=g.pageY,c.cursorAt&&this._adjustOffsetFromHelper(c.cursorAt),this.domPosition={prev:this.currentItem.prev()[0],parent:this.currentItem.parent()[0]},this.helper[0]!==this.currentItem[0]&&this.currentItem.hide(),this._createPlaceholder(),c.containment&&this._setContainment(),c.cursor&&"auto"!==c.cursor&&(h=this.document.find("body"),this.storedCursor=h.css("cursor"),h.css("cursor",c.cursor),this.storedStylesheet=k("<style>*{ cursor: "+c.cursor+" !important; }</style>").appendTo(h)),c.opacity&&(this.helper.css("opacity")&&(this._storedOpacity=this.helper.css("opacity")),this.helper.css("opacity",c.opacity)),c.zIndex&&(this.helper.css("zIndex")&&(this._storedZIndex=this.helper.css("zIndex")),this.helper.css("zIndex",c.zIndex)),this.scrollParent[0]!==this.document[0]&&"HTML"!==this.scrollParent[0].tagName&&(this.overflowOffset=this.scrollParent.offset()),this._trigger("start",g,this._uiHash()),this._preserveHelperProportions||this._cacheHelperProportions(),!f){for(l=this.containers.length-1;
l>=0;
l--){this.containers[l]._trigger("activate",g,this._uiHash(this))
}}return k.ui.ddmanager&&(k.ui.ddmanager.current=this),k.ui.ddmanager&&!c.dropBehaviour&&k.ui.ddmanager.prepareOffsets(this,g),this.dragging=!0,this._addClass(this.helper,"ui-sortable-helper"),this._mouseDrag(g),!0
},_mouseDrag:function(h){var d,f,m,l,c=this.options,g=!1;
for(this.position=this._generatePosition(h),this.positionAbs=this._convertPositionTo("absolute"),this.lastPositionAbs||(this.lastPositionAbs=this.positionAbs),this.options.scroll&&(this.scrollParent[0]!==this.document[0]&&"HTML"!==this.scrollParent[0].tagName?(this.overflowOffset.top+this.scrollParent[0].offsetHeight-h.pageY<c.scrollSensitivity?this.scrollParent[0].scrollTop=g=this.scrollParent[0].scrollTop+c.scrollSpeed:h.pageY-this.overflowOffset.top<c.scrollSensitivity&&(this.scrollParent[0].scrollTop=g=this.scrollParent[0].scrollTop-c.scrollSpeed),this.overflowOffset.left+this.scrollParent[0].offsetWidth-h.pageX<c.scrollSensitivity?this.scrollParent[0].scrollLeft=g=this.scrollParent[0].scrollLeft+c.scrollSpeed:h.pageX-this.overflowOffset.left<c.scrollSensitivity&&(this.scrollParent[0].scrollLeft=g=this.scrollParent[0].scrollLeft-c.scrollSpeed)):(h.pageY-this.document.scrollTop()<c.scrollSensitivity?g=this.document.scrollTop(this.document.scrollTop()-c.scrollSpeed):this.window.height()-(h.pageY-this.document.scrollTop())<c.scrollSensitivity&&(g=this.document.scrollTop(this.document.scrollTop()+c.scrollSpeed)),h.pageX-this.document.scrollLeft()<c.scrollSensitivity?g=this.document.scrollLeft(this.document.scrollLeft()-c.scrollSpeed):this.window.width()-(h.pageX-this.document.scrollLeft())<c.scrollSensitivity&&(g=this.document.scrollLeft(this.document.scrollLeft()+c.scrollSpeed))),g!==!1&&k.ui.ddmanager&&!c.dropBehaviour&&k.ui.ddmanager.prepareOffsets(this,h)),this.positionAbs=this._convertPositionTo("absolute"),this.options.axis&&"y"===this.options.axis||(this.helper[0].style.left=this.position.left+"px"),this.options.axis&&"x"===this.options.axis||(this.helper[0].style.top=this.position.top+"px"),d=this.items.length-1;
d>=0;
d--){if(f=this.items[d],m=f.item[0],l=this._intersectsWithPointer(f),l&&f.instance===this.currentContainer&&m!==this.currentItem[0]&&this.placeholder[1===l?"next":"prev"]()[0]!==m&&!k.contains(this.placeholder[0],m)&&("semi-dynamic"===this.options.type?!k.contains(this.element[0],m):!0)){if(this.direction=1===l?"down":"up","pointer"!==this.options.tolerance&&!this._intersectsWithSides(f)){break
}this._rearrange(h,f),this._trigger("change",h,this._uiHash());
break
}}return this._contactContainers(h),k.ui.ddmanager&&k.ui.ddmanager.drag(this,h),this._trigger("sort",h,this._uiHash()),this.lastPositionAbs=this.positionAbs,!1
},_mouseStop:function(g,d){if(g){if(k.ui.ddmanager&&!this.options.dropBehaviour&&k.ui.ddmanager.drop(this,g),this.options.revert){var f=this,l=this.placeholder.offset(),h=this.options.axis,c={};
h&&"x"!==h||(c.left=l.left-this.offset.parent.left-this.margins.left+(this.offsetParent[0]===this.document[0].body?0:this.offsetParent[0].scrollLeft)),h&&"y"!==h||(c.top=l.top-this.offset.parent.top-this.margins.top+(this.offsetParent[0]===this.document[0].body?0:this.offsetParent[0].scrollTop)),this.reverting=!0,k(this.helper).animate(c,parseInt(this.options.revert,10)||500,function(){f._clear(g)
})
}else{this._clear(g,d)
}return !1
}},cancel:function(){if(this.dragging){this._mouseUp(new k.Event("mouseup",{target:null})),"original"===this.options.helper?(this.currentItem.css(this._storedCSS),this._removeClass(this.currentItem,"ui-sortable-helper")):this.currentItem.show();
for(var a=this.containers.length-1;
a>=0;
a--){this.containers[a]._trigger("deactivate",null,this._uiHash(this)),this.containers[a].containerCache.over&&(this.containers[a]._trigger("out",null,this._uiHash(this)),this.containers[a].containerCache.over=0)
}}return this.placeholder&&(this.placeholder[0].parentNode&&this.placeholder[0].parentNode.removeChild(this.placeholder[0]),"original"!==this.options.helper&&this.helper&&this.helper[0].parentNode&&this.helper.remove(),k.extend(this,{helper:null,dragging:!1,reverting:!1,_noFinalSort:null}),this.domPosition.prev?k(this.domPosition.prev).after(this.currentItem):k(this.domPosition.parent).prepend(this.currentItem)),this
},serialize:function(d){var a=this._getItemsAsjQuery(d&&d.connected),c=[];
return d=d||{},k(a).each(function(){var e=(k(d.item||this).attr(d.attribute||"id")||"").match(d.expression||/(.+)[\-=_](.+)/);
e&&c.push((d.key||e[1]+"[]")+"="+(d.key&&d.expression?e[1]:e[2]))
}),!c.length&&d.key&&c.push(d.key+"="),c.join("&")
},toArray:function(d){var a=this._getItemsAsjQuery(d&&d.connected),c=[];
return d=d||{},a.each(function(){c.push(k(d.item||this).attr(d.attribute||"id")||"")
}),c
},_intersectsWith:function(T){var O=this.positionAbs.left,M=O+this.helperProportions.width,U=this.positionAbs.top,v=U+this.helperProportions.height,m=T.left,R=m+T.width,f=T.top,N=f+T.height,L=this.offset.click.top,Q=this.offset.click.left,S="x"===this.options.axis||U+L>f&&N>U+L,P="y"===this.options.axis||O+Q>m&&R>O+Q,g=S&&P;
return"pointer"===this.options.tolerance||this.options.forcePointerForContainers||"pointer"!==this.options.tolerance&&this.helperProportions[this.floating?"width":"height"]>T[this.floating?"width":"height"]?g:O+this.helperProportions.width/2>m&&R>M-this.helperProportions.width/2&&U+this.helperProportions.height/2>f&&N>v-this.helperProportions.height/2
},_intersectsWithPointer:function(c){var f,a,d="x"===this.options.axis||this._isOverAxis(this.positionAbs.top+this.offset.click.top,c.top,c.height),h="y"===this.options.axis||this._isOverAxis(this.positionAbs.left+this.offset.click.left,c.left,c.width),g=d&&h;
return g?(f=this._getDragVerticalDirection(),a=this._getDragHorizontalDirection(),this.floating?"right"===a||"down"===f?2:1:f&&("down"===f?2:1)):!1
},_intersectsWithSides:function(c){var f=this._isOverAxis(this.positionAbs.top+this.offset.click.top,c.top+c.height/2,c.height),a=this._isOverAxis(this.positionAbs.left+this.offset.click.left,c.left+c.width/2,c.width),d=this._getDragVerticalDirection(),g=this._getDragHorizontalDirection();
return this.floating&&g?"right"===g&&a||"left"===g&&!a:d&&("down"===d&&f||"up"===d&&!f)
},_getDragVerticalDirection:function(){var a=this.positionAbs.top-this.lastPositionAbs.top;
return 0!==a&&(a>0?"down":"up")
},_getDragHorizontalDirection:function(){var a=this.positionAbs.left-this.lastPositionAbs.left;
return 0!==a&&(a>0?"right":"left")
},refresh:function(a){return this._refreshItems(a),this._setHandleClassName(),this.refreshPositions(),this
},_connectWith:function(){var a=this.options;
return a.connectWith.constructor===String?[a.connectWith]:a.connectWith
},_getItemsAsjQuery:function(t){function m(){c.push(this)
}var v,f,d,u,c=[],p=[],g=this._connectWith();
if(g&&t){for(v=g.length-1;
v>=0;
v--){for(d=k(g[v],this.document[0]),f=d.length-1;
f>=0;
f--){u=k.data(d[f],this.widgetFullName),u&&u!==this&&!u.options.disabled&&p.push([k.isFunction(u.options.items)?u.options.items.call(u.element):k(u.options.items,u.element).not(".ui-sortable-helper").not(".ui-sortable-placeholder"),u])
}}}for(p.push([k.isFunction(this.options.items)?this.options.items.call(this.element,null,{options:this.options,item:this.currentItem}):k(this.options.items,this.element).not(".ui-sortable-helper").not(".ui-sortable-placeholder"),this]),v=p.length-1;
v>=0;
v--){p[v][0].each(m)
}return k(c)
},_removeCurrentsFromItems:function(){var a=this.currentItem.find(":data("+this.widgetName+"-item)");
this.items=k.grep(this.items,function(d){for(var c=0;
a.length>c;
c++){if(a[c]===d.item[0]){return !1
}}return !0
})
},_refreshItems:function(L){this.items=[],this.containers=[this];
var t,Q,m,g,O,f,v,p,N=this.items,P=[[k.isFunction(this.options.items)?this.options.items.call(this.element[0],L,{item:this.currentItem}):k(this.options.items,this.element),this]],M=this._connectWith();
if(M&&this.ready){for(t=M.length-1;
t>=0;
t--){for(m=k(M[t],this.document[0]),Q=m.length-1;
Q>=0;
Q--){g=k.data(m[Q],this.widgetFullName),g&&g!==this&&!g.options.disabled&&(P.push([k.isFunction(g.options.items)?g.options.items.call(g.element[0],L,{item:this.currentItem}):k(g.options.items,g.element),g]),this.containers.push(g))
}}}for(t=P.length-1;
t>=0;
t--){for(O=P[t][1],f=P[t][0],Q=0,p=f.length;
p>Q;
Q++){v=k(f[Q]),v.data(this.widgetName+"-item",O),N.push({item:v,instance:O,width:0,height:0,left:0,top:0})
}}},refreshPositions:function(d){this.floating=this.items.length?"x"===this.options.axis||this._isFloating(this.items[0].item):!1,this.offsetParent&&this.helper&&(this.offset.parent=this._getParentOffset());
var a,c,g,f;
for(a=this.items.length-1;
a>=0;
a--){c=this.items[a],c.instance!==this.currentContainer&&this.currentContainer&&c.item[0]!==this.currentItem[0]||(g=this.options.toleranceElement?k(this.options.toleranceElement,c.item):c.item,d||(c.width=g.outerWidth(),c.height=g.outerHeight()),f=g.offset(),c.left=f.left,c.top=f.top)
}if(this.options.custom&&this.options.custom.refreshContainers){this.options.custom.refreshContainers.call(this)
}else{for(a=this.containers.length-1;
a>=0;
a--){f=this.containers[a].element.offset(),this.containers[a].containerCache.left=f.left,this.containers[a].containerCache.top=f.top,this.containers[a].containerCache.width=this.containers[a].element.outerWidth(),this.containers[a].containerCache.height=this.containers[a].element.outerHeight()
}}return this
},_createPlaceholder:function(d){d=d||this;
var a,c=d.options;
c.placeholder&&c.placeholder.constructor!==String||(a=c.placeholder,c.placeholder={element:function(){var e=d.currentItem[0].nodeName.toLowerCase(),f=k("<"+e+">",d.document[0]);
return d._addClass(f,"ui-sortable-placeholder",a||d.currentItem[0].className)._removeClass(f,"ui-sortable-helper"),"tbody"===e?d._createTrPlaceholder(d.currentItem.find("tr").eq(0),k("<tr>",d.document[0]).appendTo(f)):"tr"===e?d._createTrPlaceholder(d.currentItem,f):"img"===e&&f.attr("src",d.currentItem.attr("src")),a||f.css("visibility","hidden"),f
},update:function(e,f){(!a||c.forcePlaceholderSize)&&(f.height()||f.height(d.currentItem.innerHeight()-parseInt(d.currentItem.css("paddingTop")||0,10)-parseInt(d.currentItem.css("paddingBottom")||0,10)),f.width()||f.width(d.currentItem.innerWidth()-parseInt(d.currentItem.css("paddingLeft")||0,10)-parseInt(d.currentItem.css("paddingRight")||0,10)))
}}),d.placeholder=k(c.placeholder.element.call(d.element,d.currentItem)),d.currentItem.after(d.placeholder),c.placeholder.update(d,d.placeholder)
},_createTrPlaceholder:function(d,a){var c=this;
d.children().each(function(){k("<td>&#160;</td>",c.document[0]).attr("colspan",k(this).attr("colspan")||1).appendTo(a)
})
},_contactContainers:function(N){var L,S,t,m,Q,f,M,v,P,R,O=null,g=null;
for(L=this.containers.length-1;
L>=0;
L--){if(!k.contains(this.currentItem[0],this.containers[L].element[0])){if(this._intersectsWith(this.containers[L].containerCache)){if(O&&k.contains(this.containers[L].element[0],O.element[0])){continue
}O=this.containers[L],g=L
}else{this.containers[L].containerCache.over&&(this.containers[L]._trigger("out",N,this._uiHash(this)),this.containers[L].containerCache.over=0)
}}}if(O){if(1===this.containers.length){this.containers[g].containerCache.over||(this.containers[g]._trigger("over",N,this._uiHash(this)),this.containers[g].containerCache.over=1)
}else{for(t=10000,m=null,P=O.floating||this._isFloating(this.currentItem),Q=P?"left":"top",f=P?"width":"height",R=P?"pageX":"pageY",S=this.items.length-1;
S>=0;
S--){k.contains(this.containers[g].element[0],this.items[S].item[0])&&this.items[S].item[0]!==this.currentItem[0]&&(M=this.items[S].item.offset()[Q],v=!1,N[R]-M>this.items[S][f]/2&&(v=!0),t>Math.abs(N[R]-M)&&(t=Math.abs(N[R]-M),m=this.items[S],this.direction=v?"up":"down"))
}if(!m&&!this.options.dropOnEmpty){return
}if(this.currentContainer===this.containers[g]){return this.currentContainer.containerCache.over||(this.containers[g]._trigger("over",N,this._uiHash()),this.currentContainer.containerCache.over=1),void 0
}m?this._rearrange(N,m,null,!0):this._rearrange(N,null,this.containers[g].element,!0),this._trigger("change",N,this._uiHash()),this.containers[g]._trigger("change",N,this._uiHash(this)),this.currentContainer=this.containers[g],this.options.placeholder.update(this.currentContainer,this.placeholder),this.containers[g]._trigger("over",N,this._uiHash(this)),this.containers[g].containerCache.over=1
}}},_createHelper:function(d){var a=this.options,c=k.isFunction(a.helper)?k(a.helper.apply(this.element[0],[d,this.currentItem])):"clone"===a.helper?this.currentItem.clone():this.currentItem;
return c.parents("body").length||k("parent"!==a.appendTo?a.appendTo:this.currentItem[0].parentNode)[0].appendChild(c[0]),c[0]===this.currentItem[0]&&(this._storedCSS={width:this.currentItem[0].style.width,height:this.currentItem[0].style.height,position:this.currentItem.css("position"),top:this.currentItem.css("top"),left:this.currentItem.css("left")}),(!c[0].style.width||a.forceHelperSize)&&c.width(this.currentItem.width()),(!c[0].style.height||a.forceHelperSize)&&c.height(this.currentItem.height()),c
},_adjustOffsetFromHelper:function(a){"string"==typeof a&&(a=a.split(" ")),k.isArray(a)&&(a={left:+a[0],top:+a[1]||0}),"left" in a&&(this.offset.click.left=a.left+this.margins.left),"right" in a&&(this.offset.click.left=this.helperProportions.width-a.right+this.margins.left),"top" in a&&(this.offset.click.top=a.top+this.margins.top),"bottom" in a&&(this.offset.click.top=this.helperProportions.height-a.bottom+this.margins.top)
},_getParentOffset:function(){this.offsetParent=this.helper.offsetParent();
var a=this.offsetParent.offset();
return"absolute"===this.cssPosition&&this.scrollParent[0]!==this.document[0]&&k.contains(this.scrollParent[0],this.offsetParent[0])&&(a.left+=this.scrollParent.scrollLeft(),a.top+=this.scrollParent.scrollTop()),(this.offsetParent[0]===this.document[0].body||this.offsetParent[0].tagName&&"html"===this.offsetParent[0].tagName.toLowerCase()&&k.ui.ie)&&(a={top:0,left:0}),{top:a.top+(parseInt(this.offsetParent.css("borderTopWidth"),10)||0),left:a.left+(parseInt(this.offsetParent.css("borderLeftWidth"),10)||0)}
},_getRelativeOffset:function(){if("relative"===this.cssPosition){var a=this.currentItem.position();
return{top:a.top-(parseInt(this.helper.css("top"),10)||0)+this.scrollParent.scrollTop(),left:a.left-(parseInt(this.helper.css("left"),10)||0)+this.scrollParent.scrollLeft()}
}return{top:0,left:0}
},_cacheMargins:function(){this.margins={left:parseInt(this.currentItem.css("marginLeft"),10)||0,top:parseInt(this.currentItem.css("marginTop"),10)||0}
},_cacheHelperProportions:function(){this.helperProportions={width:this.helper.outerWidth(),height:this.helper.outerHeight()}
},_setContainment:function(){var d,a,c,f=this.options;
"parent"===f.containment&&(f.containment=this.helper[0].parentNode),("document"===f.containment||"window"===f.containment)&&(this.containment=[0-this.offset.relative.left-this.offset.parent.left,0-this.offset.relative.top-this.offset.parent.top,"document"===f.containment?this.document.width():this.window.width()-this.helperProportions.width-this.margins.left,("document"===f.containment?this.document.height()||document.body.parentNode.scrollHeight:this.window.height()||this.document[0].body.parentNode.scrollHeight)-this.helperProportions.height-this.margins.top]),/^(document|window|parent)$/.test(f.containment)||(d=k(f.containment)[0],a=k(f.containment).offset(),c="hidden"!==k(d).css("overflow"),this.containment=[a.left+(parseInt(k(d).css("borderLeftWidth"),10)||0)+(parseInt(k(d).css("paddingLeft"),10)||0)-this.margins.left,a.top+(parseInt(k(d).css("borderTopWidth"),10)||0)+(parseInt(k(d).css("paddingTop"),10)||0)-this.margins.top,a.left+(c?Math.max(d.scrollWidth,d.offsetWidth):d.offsetWidth)-(parseInt(k(d).css("borderLeftWidth"),10)||0)-(parseInt(k(d).css("paddingRight"),10)||0)-this.helperProportions.width-this.margins.left,a.top+(c?Math.max(d.scrollHeight,d.offsetHeight):d.offsetHeight)-(parseInt(k(d).css("borderTopWidth"),10)||0)-(parseInt(k(d).css("paddingBottom"),10)||0)-this.helperProportions.height-this.margins.top])
},_convertPositionTo:function(d,a){a||(a=this.position);
var c="absolute"===d?1:-1,g="absolute"!==this.cssPosition||this.scrollParent[0]!==this.document[0]&&k.contains(this.scrollParent[0],this.offsetParent[0])?this.scrollParent:this.offsetParent,f=/(html|body)/i.test(g[0].tagName);
return{top:a.top+this.offset.relative.top*c+this.offset.parent.top*c-("fixed"===this.cssPosition?-this.scrollParent.scrollTop():f?0:g.scrollTop())*c,left:a.left+this.offset.relative.left*c+this.offset.parent.left*c-("fixed"===this.cssPosition?-this.scrollParent.scrollLeft():f?0:g.scrollLeft())*c}
},_generatePosition:function(m){var d,g,t=this.options,p=m.pageX,c=m.pageY,l="absolute"!==this.cssPosition||this.scrollParent[0]!==this.document[0]&&k.contains(this.scrollParent[0],this.offsetParent[0])?this.scrollParent:this.offsetParent,f=/(html|body)/i.test(l[0].tagName);
return"relative"!==this.cssPosition||this.scrollParent[0]!==this.document[0]&&this.scrollParent[0]!==this.offsetParent[0]||(this.offset.relative=this._getRelativeOffset()),this.originalPosition&&(this.containment&&(m.pageX-this.offset.click.left<this.containment[0]&&(p=this.containment[0]+this.offset.click.left),m.pageY-this.offset.click.top<this.containment[1]&&(c=this.containment[1]+this.offset.click.top),m.pageX-this.offset.click.left>this.containment[2]&&(p=this.containment[2]+this.offset.click.left),m.pageY-this.offset.click.top>this.containment[3]&&(c=this.containment[3]+this.offset.click.top)),t.grid&&(d=this.originalPageY+Math.round((c-this.originalPageY)/t.grid[1])*t.grid[1],c=this.containment?d-this.offset.click.top>=this.containment[1]&&d-this.offset.click.top<=this.containment[3]?d:d-this.offset.click.top>=this.containment[1]?d-t.grid[1]:d+t.grid[1]:d,g=this.originalPageX+Math.round((p-this.originalPageX)/t.grid[0])*t.grid[0],p=this.containment?g-this.offset.click.left>=this.containment[0]&&g-this.offset.click.left<=this.containment[2]?g:g-this.offset.click.left>=this.containment[0]?g-t.grid[0]:g+t.grid[0]:g)),{top:c-this.offset.click.top-this.offset.relative.top-this.offset.parent.top+("fixed"===this.cssPosition?-this.scrollParent.scrollTop():f?0:l.scrollTop()),left:p-this.offset.click.left-this.offset.relative.left-this.offset.parent.left+("fixed"===this.cssPosition?-this.scrollParent.scrollLeft():f?0:l.scrollLeft())}
},_rearrange:function(c,f,a,d){a?a[0].appendChild(this.placeholder[0]):f.item[0].parentNode.insertBefore(this.placeholder[0],"down"===this.direction?f.item[0]:f.item[0].nextSibling),this.counter=this.counter?++this.counter:1;
var g=this.counter;
this._delay(function(){g===this.counter&&this.refreshPositions(!d)
})
},_clear:function(c,f){function a(l,m,h){return function(e){h._trigger(l,e,m._uiHash(m))
}
}this.reverting=!1;
var d,g=[];
if(!this._noFinalSort&&this.currentItem.parent().length&&this.placeholder.before(this.currentItem),this._noFinalSort=null,this.helper[0]===this.currentItem[0]){for(d in this._storedCSS){("auto"===this._storedCSS[d]||"static"===this._storedCSS[d])&&(this._storedCSS[d]="")
}this.currentItem.css(this._storedCSS),this._removeClass(this.currentItem,"ui-sortable-helper")
}else{this.currentItem.show()
}for(this.fromOutside&&!f&&g.push(function(e){this._trigger("receive",e,this._uiHash(this.fromOutside))
}),!this.fromOutside&&this.domPosition.prev===this.currentItem.prev().not(".ui-sortable-helper")[0]&&this.domPosition.parent===this.currentItem.parent()[0]||f||g.push(function(e){this._trigger("update",e,this._uiHash())
}),this!==this.currentContainer&&(f||(g.push(function(e){this._trigger("remove",e,this._uiHash())
}),g.push(function(e){return function(h){e._trigger("receive",h,this._uiHash(this))
}
}.call(this,this.currentContainer)),g.push(function(e){return function(h){e._trigger("update",h,this._uiHash(this))
}
}.call(this,this.currentContainer)))),d=this.containers.length-1;
d>=0;
d--){f||g.push(a("deactivate",this,this.containers[d])),this.containers[d].containerCache.over&&(g.push(a("out",this,this.containers[d])),this.containers[d].containerCache.over=0)
}if(this.storedCursor&&(this.document.find("body").css("cursor",this.storedCursor),this.storedStylesheet.remove()),this._storedOpacity&&this.helper.css("opacity",this._storedOpacity),this._storedZIndex&&this.helper.css("zIndex","auto"===this._storedZIndex?"":this._storedZIndex),this.dragging=!1,f||this._trigger("beforeStop",c,this._uiHash()),this.placeholder[0].parentNode.removeChild(this.placeholder[0]),this.cancelHelperRemoval||(this.helper[0]!==this.currentItem[0]&&this.helper.remove(),this.helper=null),!f){for(d=0;
g.length>d;
d++){g[d].call(this,c)
}this._trigger("stop",c,this._uiHash())
}return this.fromOutside=!1,!this.cancelHelperRemoval
},_trigger:function(){k.Widget.prototype._trigger.apply(this,arguments)===!1&&this.cancel()
},_uiHash:function(c){var a=c||this;
return{helper:a.helper,placeholder:a.placeholder||k([]),position:a.position,originalPosition:a.originalPosition,offset:a.positionAbs,item:a.currentItem,sender:c?c.element:null}
}}),k.widget("ui.spinner",{version:"1.12.1",defaultElement:"<input>",widgetEventPrefix:"spin",options:{classes:{"ui-spinner":"ui-corner-all","ui-spinner-down":"ui-corner-br","ui-spinner-up":"ui-corner-tr"},culture:null,icons:{down:"ui-icon-triangle-1-s",up:"ui-icon-triangle-1-n"},incremental:!0,max:null,min:null,numberFormat:null,page:10,step:1,change:null,spin:null,start:null,stop:null},_create:function(){this._setOption("max",this.options.max),this._setOption("min",this.options.min),this._setOption("step",this.options.step),""!==this.value()&&this._value(this.element.val(),!0),this._draw(),this._on(this._events),this._refresh(),this._on(this.window,{beforeunload:function(){this.element.removeAttr("autocomplete")
}})
},_getCreateOptions:function(){var c=this._super(),a=this.element;
return k.each(["min","max","step"],function(d,e){var f=a.attr(e);
null!=f&&f.length&&(c[e]=f)
}),c
},_events:{keydown:function(a){this._start(a)&&this._keydown(a)&&a.preventDefault()
},keyup:"_stop",focus:function(){this.previous=this.element.val()
},blur:function(a){return this.cancelBlur?(delete this.cancelBlur,void 0):(this._stop(),this._refresh(),this.previous!==this.element.val()&&this._trigger("change",a),void 0)
},mousewheel:function(a,c){if(c){if(!this.spinning&&!this._start(a)){return !1
}this._spin((c>0?1:-1)*this.options.step,a),clearTimeout(this.mousewheelTimer),this.mousewheelTimer=this._delay(function(){this.spinning&&this._stop(a)
},100),a.preventDefault()
}},"mousedown .ui-spinner-button":function(d){function a(){var f=this.element[0]===k.ui.safeActiveElement(this.document[0]);
f||(this.element.trigger("focus"),this.previous=c,this._delay(function(){this.previous=c
}))
}var c;
c=this.element[0]===k.ui.safeActiveElement(this.document[0])?this.previous:this.element.val(),d.preventDefault(),a.call(this),this.cancelBlur=!0,this._delay(function(){delete this.cancelBlur,a.call(this)
}),this._start(d)!==!1&&this._repeat(null,k(d.currentTarget).hasClass("ui-spinner-up")?1:-1,d)
},"mouseup .ui-spinner-button":"_stop","mouseenter .ui-spinner-button":function(a){return k(a.currentTarget).hasClass("ui-state-active")?this._start(a)===!1?!1:(this._repeat(null,k(a.currentTarget).hasClass("ui-spinner-up")?1:-1,a),void 0):void 0
},"mouseleave .ui-spinner-button":"_stop"},_enhance:function(){this.uiSpinner=this.element.attr("autocomplete","off").wrap("<span>").parent().append("<a></a><a></a>")
},_draw:function(){this._enhance(),this._addClass(this.uiSpinner,"ui-spinner","ui-widget ui-widget-content"),this._addClass("ui-spinner-input"),this.element.attr("role","spinbutton"),this.buttons=this.uiSpinner.children("a").attr("tabIndex",-1).attr("aria-hidden",!0).button({classes:{"ui-button":""}}),this._removeClass(this.buttons,"ui-corner-all"),this._addClass(this.buttons.first(),"ui-spinner-button ui-spinner-up"),this._addClass(this.buttons.last(),"ui-spinner-button ui-spinner-down"),this.buttons.first().button({icon:this.options.icons.up,showLabel:!1}),this.buttons.last().button({icon:this.options.icons.down,showLabel:!1}),this.buttons.height()>Math.ceil(0.5*this.uiSpinner.height())&&this.uiSpinner.height()>0&&this.uiSpinner.height(this.uiSpinner.height())
},_keydown:function(d){var a=this.options,c=k.ui.keyCode;
switch(d.keyCode){case c.UP:return this._repeat(null,1,d),!0;
case c.DOWN:return this._repeat(null,-1,d),!0;
case c.PAGE_UP:return this._repeat(null,a.page,d),!0;
case c.PAGE_DOWN:return this._repeat(null,-a.page,d),!0
}return !1
},_start:function(a){return this.spinning||this._trigger("start",a)!==!1?(this.counter||(this.counter=1),this.spinning=!0,!0):!1
},_repeat:function(c,d,a){c=c||500,clearTimeout(this.timer),this.timer=this._delay(function(){this._repeat(40,d,a)
},c),this._spin(d*this.options.step,a)
},_spin:function(c,d){var a=this.value()||0;
this.counter||(this.counter=1),a=this._adjustValue(a+c*this._increment(this.counter)),this.spinning&&this._trigger("spin",d,{value:a})===!1||(this._value(a),this.counter++)
},_increment:function(c){var a=this.options.incremental;
return a?k.isFunction(a)?a(c):Math.floor(c*c*c/50000-c*c/500+17*c/200+1):1
},_precision:function(){var a=this._precisionOf(this.options.step);
return null!==this.options.min&&(a=Math.max(a,this._precisionOf(this.options.min))),a
},_precisionOf:function(c){var d=""+c,a=d.indexOf(".");
return -1===a?0:d.length-a-1
},_adjustValue:function(c){var f,a,d=this.options;
return f=null!==d.min?d.min:0,a=c-f,a=Math.round(a/d.step)*d.step,c=f+a,c=parseFloat(c.toFixed(this._precision())),null!==d.max&&c>d.max?d.max:null!==d.min&&d.min>c?d.min:c
},_stop:function(a){this.spinning&&(clearTimeout(this.timer),clearTimeout(this.mousewheelTimer),this.counter=0,this.spinning=!1,this._trigger("stop",a))
},_setOption:function(c,f){var a,d,g;
return"culture"===c||"numberFormat"===c?(a=this._parse(this.element.val()),this.options[c]=f,this.element.val(this._format(a)),void 0):(("max"===c||"min"===c||"step"===c)&&"string"==typeof f&&(f=this._parse(f)),"icons"===c&&(d=this.buttons.first().find(".ui-icon"),this._removeClass(d,null,this.options.icons.up),this._addClass(d,null,f.up),g=this.buttons.last().find(".ui-icon"),this._removeClass(g,null,this.options.icons.down),this._addClass(g,null,f.down)),this._super(c,f),void 0)
},_setOptionDisabled:function(a){this._super(a),this._toggleClass(this.uiSpinner,null,"ui-state-disabled",!!a),this.element.prop("disabled",!!a),this.buttons.button(a?"disable":"enable")
},_setOptions:w(function(a){this._super(a)
}),_parse:function(a){return"string"==typeof a&&""!==a&&(a=window.Globalize&&this.options.numberFormat?Globalize.parseFloat(a,10,this.options.culture):+a),""===a||isNaN(a)?null:a
},_format:function(a){return""===a?"":window.Globalize&&this.options.numberFormat?Globalize.format(a,this.options.numberFormat,this.options.culture):a
},_refresh:function(){this.element.attr({"aria-valuemin":this.options.min,"aria-valuemax":this.options.max,"aria-valuenow":this._parse(this.element.val())})
},isValid:function(){var a=this.value();
return null===a?!1:a===this._adjustValue(a)
},_value:function(c,d){var a;
""!==c&&(a=this._parse(c),null!==a&&(d||(a=this._adjustValue(a)),c=this._format(a))),this.element.val(c),this._refresh()
},_destroy:function(){this.element.prop("disabled",!1).removeAttr("autocomplete role aria-valuemin aria-valuemax aria-valuenow"),this.uiSpinner.replaceWith(this.element)
},stepUp:w(function(a){this._stepUp(a)
}),_stepUp:function(a){this._start()&&(this._spin((a||1)*this.options.step),this._stop())
},stepDown:w(function(a){this._stepDown(a)
}),_stepDown:function(a){this._start()&&(this._spin((a||1)*-this.options.step),this._stop())
},pageUp:w(function(a){this._stepUp((a||1)*this.options.page)
}),pageDown:w(function(a){this._stepDown((a||1)*this.options.page)
}),value:function(a){return arguments.length?(w(this._value).call(this,a),void 0):this._parse(this.element.val())
},widget:function(){return this.uiSpinner
}}),k.uiBackCompat!==!1&&k.widget("ui.spinner",k.ui.spinner,{_enhance:function(){this.uiSpinner=this.element.attr("autocomplete","off").wrap(this._uiSpinnerHtml()).parent().append(this._buttonHtml())
},_uiSpinnerHtml:function(){return"<span>"
},_buttonHtml:function(){return"<a></a><a></a>"
}}),k.ui.spinner,k.widget("ui.tabs",{version:"1.12.1",delay:300,options:{active:null,classes:{"ui-tabs":"ui-corner-all","ui-tabs-nav":"ui-corner-all","ui-tabs-panel":"ui-corner-bottom","ui-tabs-tab":"ui-corner-top"},collapsible:!1,event:"click",heightStyle:"content",hide:null,show:null,activate:null,beforeActivate:null,beforeLoad:null,load:null},_isLocal:function(){var a=/#.*$/;
return function(f){var c,d;
c=f.href.replace(a,""),d=location.href.replace(a,"");
try{c=decodeURIComponent(c)
}catch(g){}try{d=decodeURIComponent(d)
}catch(g){}return f.hash.length>1&&c===d
}
}(),_create:function(){var c=this,a=this.options;
this.running=!1,this._addClass("ui-tabs","ui-widget ui-widget-content"),this._toggleClass("ui-tabs-collapsible",null,a.collapsible),this._processTabs(),a.active=this._initialActive(),k.isArray(a.disabled)&&(a.disabled=k.unique(a.disabled.concat(k.map(this.tabs.filter(".ui-state-disabled"),function(d){return c.tabs.index(d)
}))).sort()),this.active=this.options.active!==!1&&this.anchors.length?this._findActive(a.active):k(),this._refresh(),this.active.length&&this.load(a.active)
},_initialActive:function(){var d=this.options.active,a=this.options.collapsible,c=location.hash.substring(1);
return null===d&&(c&&this.tabs.each(function(e,f){return k(f).attr("aria-controls")===c?(d=e,!1):void 0
}),null===d&&(d=this.tabs.index(this.tabs.filter(".ui-tabs-active"))),(null===d||-1===d)&&(d=this.tabs.length?0:!1)),d!==!1&&(d=this.tabs.index(this.tabs.eq(d)),-1===d&&(d=a?!1:0)),!a&&d===!1&&this.anchors.length&&(d=0),d
},_getCreateEventData:function(){return{tab:this.active,panel:this.active.length?this._getPanelForTab(this.active):k()}
},_tabKeydown:function(d){var a=k(k.ui.safeActiveElement(this.document[0])).closest("li"),c=this.tabs.index(a),f=!0;
if(!this._handlePageNav(d)){switch(d.keyCode){case k.ui.keyCode.RIGHT:case k.ui.keyCode.DOWN:c++;
break;
case k.ui.keyCode.UP:case k.ui.keyCode.LEFT:f=!1,c--;
break;
case k.ui.keyCode.END:c=this.anchors.length-1;
break;
case k.ui.keyCode.HOME:c=0;
break;
case k.ui.keyCode.SPACE:return d.preventDefault(),clearTimeout(this.activating),this._activate(c),void 0;
case k.ui.keyCode.ENTER:return d.preventDefault(),clearTimeout(this.activating),this._activate(c===this.options.active?!1:c),void 0;
default:return
}d.preventDefault(),clearTimeout(this.activating),c=this._focusNextTab(c,f),d.ctrlKey||d.metaKey||(a.attr("aria-selected","false"),this.tabs.eq(c).attr("aria-selected","true"),this.activating=this._delay(function(){this.option("active",c)
},this.delay))
}},_panelKeydown:function(a){this._handlePageNav(a)||a.ctrlKey&&a.keyCode===k.ui.keyCode.UP&&(a.preventDefault(),this.active.trigger("focus"))
},_handlePageNav:function(a){return a.altKey&&a.keyCode===k.ui.keyCode.PAGE_UP?(this._activate(this._focusNextTab(this.options.active-1,!1)),!0):a.altKey&&a.keyCode===k.ui.keyCode.PAGE_DOWN?(this._activate(this._focusNextTab(this.options.active+1,!0)),!0):void 0
},_findNextTab:function(d,a){function c(){return d>f&&(d=0),0>d&&(d=f),d
}for(var f=this.tabs.length-1;
-1!==k.inArray(c(),this.options.disabled);
){d=a?d+1:d-1
}return d
},_focusNextTab:function(a,c){return a=this._findNextTab(a,c),this.tabs.eq(a).trigger("focus"),a
},_setOption:function(a,c){return"active"===a?(this._activate(c),void 0):(this._super(a,c),"collapsible"===a&&(this._toggleClass("ui-tabs-collapsible",null,c),c||this.options.active!==!1||this._activate(0)),"event"===a&&this._setupEvents(c),"heightStyle"===a&&this._setupHeightStyle(c),void 0)
},_sanitizeSelector:function(a){return a?a.replace(/[!"$%&'()*+,.\/:;<=>?@\[\]\^`{|}~]/g,"\\$&"):""
},refresh:function(){var c=this.options,a=this.tablist.children(":has(a[href])");
c.disabled=k.map(a.filter(".ui-state-disabled"),function(d){return a.index(d)
}),this._processTabs(),c.active!==!1&&this.anchors.length?this.active.length&&!k.contains(this.tablist[0],this.active[0])?this.tabs.length===c.disabled.length?(c.active=!1,this.active=k()):this._activate(this._findNextTab(Math.max(0,c.active-1),!1)):c.active=this.tabs.index(this.active):(c.active=!1,this.active=k()),this._refresh()
},_refresh:function(){this._setOptionDisabled(this.options.disabled),this._setupEvents(this.options.event),this._setupHeightStyle(this.options.heightStyle),this.tabs.not(this.active).attr({"aria-selected":"false","aria-expanded":"false",tabIndex:-1}),this.panels.not(this._getPanelForTab(this.active)).hide().attr({"aria-hidden":"true"}),this.active.length?(this.active.attr({"aria-selected":"true","aria-expanded":"true",tabIndex:0}),this._addClass(this.active,"ui-tabs-active","ui-state-active"),this._getPanelForTab(this.active).show().attr({"aria-hidden":"false"})):this.tabs.eq(0).attr("tabIndex",0)
},_processTabs:function(){var d=this,a=this.tabs,c=this.anchors,f=this.panels;
this.tablist=this._getList().attr("role","tablist"),this._addClass(this.tablist,"ui-tabs-nav","ui-helper-reset ui-helper-clearfix ui-widget-header"),this.tablist.on("mousedown"+this.eventNamespace,"> li",function(g){k(this).is(".ui-state-disabled")&&g.preventDefault()
}).on("focus"+this.eventNamespace,".ui-tabs-anchor",function(){k(this).closest("li").is(".ui-state-disabled")&&this.blur()
}),this.tabs=this.tablist.find("> li:has(a[href])").attr({role:"tab",tabIndex:-1}),this._addClass(this.tabs,"ui-tabs-tab","ui-state-default"),this.anchors=this.tabs.map(function(){return k("a",this)[0]
}).attr({role:"presentation",tabIndex:-1}),this._addClass(this.anchors,"ui-tabs-anchor"),this.panels=k(),this.anchors.each(function(m,t){var L,v,g,u=k(t).uniqueId().attr("id"),p=k(t).closest("li"),e=p.attr("aria-controls");
d._isLocal(t)?(L=t.hash,g=L.substring(1),v=d.element.find(d._sanitizeSelector(L))):(g=p.attr("aria-controls")||k({}).uniqueId()[0].id,L="#"+g,v=d.element.find(L),v.length||(v=d._createPanel(g),v.insertAfter(d.panels[m-1]||d.tablist)),v.attr("aria-live","polite")),v.length&&(d.panels=d.panels.add(v)),e&&p.data("ui-tabs-aria-controls",e),p.attr({"aria-controls":g,"aria-labelledby":u}),v.attr("aria-labelledby",u)
}),this.panels.attr("role","tabpanel"),this._addClass(this.panels,"ui-tabs-panel","ui-widget-content"),a&&(this._off(a.not(this.tabs)),this._off(c.not(this.anchors)),this._off(f.not(this.panels)))
},_getList:function(){return this.tablist||this.element.find("ol, ul").eq(0)
},_createPanel:function(a){return k("<div>").attr("id",a).data("ui-tabs-destroy",!0)
},_setOptionDisabled:function(d){var a,c,f;
for(k.isArray(d)&&(d.length?d.length===this.anchors.length&&(d=!0):d=!1),f=0;
c=this.tabs[f];
f++){a=k(c),d===!0||-1!==k.inArray(f,d)?(a.attr("aria-disabled","true"),this._addClass(a,null,"ui-state-disabled")):(a.removeAttr("aria-disabled"),this._removeClass(a,null,"ui-state-disabled"))
}this.options.disabled=d,this._toggleClass(this.widget(),this.widgetFullName+"-disabled",null,d===!0)
},_setupEvents:function(c){var a={};
c&&k.each(c.split(" "),function(d,f){a[f]="_eventHandler"
}),this._off(this.anchors.add(this.tabs).add(this.panels)),this._on(!0,this.anchors,{click:function(d){d.preventDefault()
}}),this._on(this.anchors,a),this._on(this.tabs,{keydown:"_tabKeydown"}),this._on(this.panels,{keydown:"_panelKeydown"}),this._focusable(this.tabs),this._hoverable(this.tabs)
},_setupHeightStyle:function(d){var a,c=this.element.parent();
"fill"===d?(a=c.height(),a-=this.element.outerHeight()-this.element.height(),this.element.siblings(":visible").each(function(){var g=k(this),f=g.css("position");
"absolute"!==f&&"fixed"!==f&&(a-=g.outerHeight(!0))
}),this.element.children().not(this.panels).each(function(){a-=k(this).outerHeight(!0)
}),this.panels.each(function(){k(this).height(Math.max(0,a-k(this).innerHeight()+k(this).height()))
}).css("overflow","auto")):"auto"===d&&(a=0,this.panels.each(function(){a=Math.max(a,k(this).height("").height())
}).height(a))
},_eventHandler:function(u){var p=this.options,M=this.active,g=k(u.currentTarget),f=g.closest("li"),L=f[0]===M[0],d=L&&p.collapsible,t=d?k():this._getPanelForTab(f),m=M.length?this._getPanelForTab(M):k(),v={oldTab:M,oldPanel:m,newTab:d?k():f,newPanel:t};
u.preventDefault(),f.hasClass("ui-state-disabled")||f.hasClass("ui-tabs-loading")||this.running||L&&!p.collapsible||this._trigger("beforeActivate",u,v)===!1||(p.active=d?!1:this.tabs.index(f),this.active=L?k():f,this.xhr&&this.xhr.abort(),m.length||t.length||k.error("jQuery UI Tabs: Mismatching fragment identifier."),t.length&&this.load(this.tabs.index(f),u),this._toggle(u,v))
},_toggle:function(h,d){function f(){l.running=!1,l._trigger("activate",h,d)
}function m(){l._addClass(d.newTab.closest("li"),"ui-tabs-active","ui-state-active"),c.length&&l.options.show?l._show(c,l.options.show,f):(c.show(),f())
}var l=this,c=d.newPanel,g=d.oldPanel;
this.running=!0,g.length&&this.options.hide?this._hide(g,this.options.hide,function(){l._removeClass(d.oldTab.closest("li"),"ui-tabs-active","ui-state-active"),m()
}):(this._removeClass(d.oldTab.closest("li"),"ui-tabs-active","ui-state-active"),g.hide(),m()),g.attr("aria-hidden","true"),d.oldTab.attr({"aria-selected":"false","aria-expanded":"false"}),c.length&&g.length?d.oldTab.attr("tabIndex",-1):c.length&&this.tabs.filter(function(){return 0===k(this).attr("tabIndex")
}).attr("tabIndex",-1),c.attr("aria-hidden","false"),d.newTab.attr({"aria-selected":"true","aria-expanded":"true",tabIndex:0})
},_activate:function(d){var a,c=this._findActive(d);
c[0]!==this.active[0]&&(c.length||(c=this.active),a=c.find(".ui-tabs-anchor")[0],this._eventHandler({target:a,currentTarget:a,preventDefault:k.noop}))
},_findActive:function(a){return a===!1?k():this.tabs.eq(a)
},_getIndex:function(a){return"string"==typeof a&&(a=this.anchors.index(this.anchors.filter("[href$='"+k.ui.escapeSelector(a)+"']"))),a
},_destroy:function(){this.xhr&&this.xhr.abort(),this.tablist.removeAttr("role").off(this.eventNamespace),this.anchors.removeAttr("role tabIndex").removeUniqueId(),this.tabs.add(this.panels).each(function(){k.data(this,"ui-tabs-destroy")?k(this).remove():k(this).removeAttr("role tabIndex aria-live aria-busy aria-selected aria-labelledby aria-hidden aria-expanded")
}),this.tabs.each(function(){var c=k(this),a=c.data("ui-tabs-aria-controls");
a?c.attr("aria-controls",a).removeData("ui-tabs-aria-controls"):c.removeAttr("aria-controls")
}),this.panels.show(),"content"!==this.options.heightStyle&&this.panels.css("height","")
},enable:function(c){var a=this.options.disabled;
a!==!1&&(void 0===c?a=!1:(c=this._getIndex(c),a=k.isArray(a)?k.map(a,function(d){return d!==c?d:null
}):k.map(this.tabs,function(e,d){return d!==c?d:null
})),this._setOptionDisabled(a))
},disable:function(c){var a=this.options.disabled;
if(a!==!0){if(void 0===c){a=!0
}else{if(c=this._getIndex(c),-1!==k.inArray(c,a)){return
}a=k.isArray(a)?k.merge([c],a).sort():[c]
}this._setOptionDisabled(a)
}},load:function(m,d){m=this._getIndex(m);
var g=this,t=this.tabs.eq(m),p=t.find(".ui-tabs-anchor"),c=this._getPanelForTab(t),l={tab:t,panel:c},f=function(a,h){"abort"===h&&g.panels.stop(!1,!0),g._removeClass(t,"ui-tabs-loading"),c.removeAttr("aria-busy"),a===g.xhr&&delete g.xhr
};
this._isLocal(p[0])||(this.xhr=k.ajax(this._ajaxSettings(p,d,l)),this.xhr&&"canceled"!==this.xhr.statusText&&(this._addClass(t,"ui-tabs-loading"),c.attr("aria-busy","true"),this.xhr.done(function(a,h,i){setTimeout(function(){c.html(a),g._trigger("load",d,l),f(i,h)
},1)
}).fail(function(a,h){setTimeout(function(){f(a,h)
},1)
})))
},_ajaxSettings:function(d,a,c){var f=this;
return{url:d.attr("href").replace(/#.*$/,""),beforeSend:function(g,h){return f._trigger("beforeLoad",a,k.extend({jqXHR:g,ajaxSettings:h},c))
}}
},_getPanelForTab:function(c){var a=k(c).attr("aria-controls");
return this.element.find(this._sanitizeSelector("#"+a))
}}),k.uiBackCompat!==!1&&k.widget("ui.tabs",k.ui.tabs,{_processTabs:function(){this._superApply(arguments),this._addClass(this.tabs,"ui-tab")
}}),k.ui.tabs,k.widget("ui.tooltip",{version:"1.12.1",options:{classes:{"ui-tooltip":"ui-corner-all ui-widget-shadow"},content:function(){var a=k(this).attr("title")||"";
return k("<a>").text(a).html()
},hide:!0,items:"[title]:not([disabled])",position:{my:"left top+15",at:"left bottom",collision:"flipfit flip"},show:!0,track:!1,close:null,open:null},_addDescribedBy:function(d,a){var c=(d.attr("aria-describedby")||"").split(/\s+/);
c.push(a),d.data("ui-tooltip-id",a).attr("aria-describedby",k.trim(c.join(" ")))
},_removeDescribedBy:function(d){var a=d.data("ui-tooltip-id"),c=(d.attr("aria-describedby")||"").split(/\s+/),f=k.inArray(a,c);
-1!==f&&c.splice(f,1),d.removeData("ui-tooltip-id"),c=k.trim(c.join(" ")),c?d.attr("aria-describedby",c):d.removeAttr("aria-describedby")
},_create:function(){this._on({mouseover:"open",focusin:"open"}),this.tooltips={},this.parents={},this.liveRegion=k("<div>").attr({role:"log","aria-live":"assertive","aria-relevant":"additions"}).appendTo(this.document[0].body),this._addClass(this.liveRegion,null,"ui-helper-hidden-accessible"),this.disabledTitles=k([])
},_setOption:function(d,a){var c=this;
this._super(d,a),"content"===d&&k.each(this.tooltips,function(f,g){c._updateContent(g.element)
})
},_setOptionDisabled:function(a){this[a?"_disable":"_enable"]()
},_disable:function(){var a=this;
k.each(this.tooltips,function(c,d){var e=k.Event("blur");
e.target=e.currentTarget=d.element[0],a.close(e,!0)
}),this.disabledTitles=this.disabledTitles.add(this.element.find(this.options.items).addBack().filter(function(){var c=k(this);
return c.is("[title]")?c.data("ui-tooltip-title",c.attr("title")).removeAttr("title"):void 0
}))
},_enable:function(){this.disabledTitles.each(function(){var a=k(this);
a.data("ui-tooltip-title")&&a.attr("title",a.data("ui-tooltip-title"))
}),this.disabledTitles=k([])
},open:function(d){var a=this,c=k(d?d.target:this.element).closest(this.options.items);
c.length&&!c.data("ui-tooltip-id")&&(c.attr("title")&&c.data("ui-tooltip-title",c.attr("title")),c.data("ui-tooltip-open",!0),d&&"mouseover"===d.type&&c.parents().each(function(){var g,f=k(this);
f.data("ui-tooltip-open")&&(g=k.Event("blur"),g.target=g.currentTarget=this,a.close(g,!0)),f.attr("title")&&(f.uniqueId(),a.parents[this.id]={element:this,title:f.attr("title")},f.attr("title",""))
}),this._registerCloseHandlers(d,c),this._updateContent(c,d))
},_updateContent:function(c,f){var a,d=this.options.content,h=this,g=f?f.type:null;
return"string"==typeof d||d.nodeType||d.jquery?this._open(f,c,d):(a=d.call(c[0],function(e){h._delay(function(){c.data("ui-tooltip-open")&&(f&&(f.type=g),this._open(f,c,e))
})
}),a&&this._open(f,c,a),void 0)
},_open:function(t,m,v){function f(a){g.of=a,u.is(":hidden")||u.position(g)
}var d,u,c,p,g=k.extend({},this.options.position);
if(v){if(d=this._find(m)){return d.tooltip.find(".ui-tooltip-content").html(v),void 0
}m.is("[title]")&&(t&&"mouseover"===t.type?m.attr("title",""):m.removeAttr("title")),d=this._tooltip(m),u=d.tooltip,this._addDescribedBy(m,u.attr("id")),u.find(".ui-tooltip-content").html(v),this.liveRegion.children().hide(),p=k("<div>").html(u.find(".ui-tooltip-content").html()),p.removeAttr("name").find("[name]").removeAttr("name"),p.removeAttr("id").find("[id]").removeAttr("id"),p.appendTo(this.liveRegion),this.options.track&&t&&/^mouse/.test(t.type)?(this._on(this.document,{mousemove:f}),f(t)):u.position(k.extend({of:m},this.options.position)),u.hide(),this._show(u,this.options.show),this.options.track&&this.options.show&&this.options.show.delay&&(c=this.delayedShow=setInterval(function(){u.is(":visible")&&(f(g.of),clearInterval(c))
},k.fx.interval)),this._trigger("open",t,{tooltip:u})
}},_registerCloseHandlers:function(d,a){var c={keyup:function(g){if(g.keyCode===k.ui.keyCode.ESCAPE){var f=k.Event(g);
f.currentTarget=a[0],this.close(f,!0)
}}};
a[0]!==this.element[0]&&(c.remove=function(){this._removeTooltip(this._find(a).tooltip)
}),d&&"mouseover"!==d.type||(c.mouseleave="close"),d&&"focusin"!==d.type||(c.focusout="close"),this._on(!0,a,c)
},close:function(d){var a,c=this,g=k(d?d.currentTarget:this.element),f=this._find(g);
return f?(a=f.tooltip,f.closing||(clearInterval(this.delayedShow),g.data("ui-tooltip-title")&&!g.attr("title")&&g.attr("title",g.data("ui-tooltip-title")),this._removeDescribedBy(g),f.hiding=!0,a.stop(!0),this._hide(a,this.options.hide,function(){c._removeTooltip(k(this))
}),g.removeData("ui-tooltip-open"),this._off(g,"mouseleave focusout keyup"),g[0]!==this.element[0]&&this._off(g,"remove"),this._off(this.document,"mousemove"),d&&"mouseleave"===d.type&&k.each(this.parents,function(l,h){k(h.element).attr("title",h.title),delete c.parents[l]
}),f.closing=!0,this._trigger("close",d,{tooltip:a}),f.hiding||(f.closing=!1)),void 0):(g.removeData("ui-tooltip-open"),void 0)
},_tooltip:function(d){var a=k("<div>").attr("role","tooltip"),c=k("<div>").appendTo(a),f=a.uniqueId().attr("id");
return this._addClass(c,"ui-tooltip-content"),this._addClass(a,"ui-tooltip","ui-widget ui-widget-content"),a.appendTo(this._appendTo(d)),this.tooltips[f]={element:d,tooltip:a}
},_find:function(a){var c=a.data("ui-tooltip-id");
return c?this.tooltips[c]:null
},_removeTooltip:function(a){a.remove(),delete this.tooltips[a.attr("id")]
},_appendTo:function(a){var c=a.closest(".ui-front, dialog");
return c.length||(c=this.document[0].body),c
},_destroy:function(){var a=this;
k.each(this.tooltips,function(c,d){var f=k.Event("blur"),e=d.element;
f.target=f.currentTarget=e[0],a.close(f,!0),k("#"+c).remove(),e.data("ui-tooltip-title")&&(e.attr("title")||e.attr("title",e.data("ui-tooltip-title")),e.removeData("ui-tooltip-title"))
}),this.liveRegion.remove()
}}),k.uiBackCompat!==!1&&k.widget("ui.tooltip",k.ui.tooltip,{options:{tooltipClass:null},_tooltip:function(){var a=this._superApply(arguments);
return this.options.tooltipClass&&a.tooltip.addClass(this.options.tooltipClass),a
}}),k.ui.tooltip
});
/*! jQuery & Zepto Lazy v1.7.7 - http://jquery.eisbehr.de/lazy - MIT&GPL-2.0 license - Copyright 2012-2017 Daniel 'Eisbehr' Kern */
;
!function(d,g){function f(S,ag,Z,K,Y){function ac(){o=d.devicePixelRatio>1,Z=ae(Z),ag.delay>=0&&setTimeout(function(){Q(!0)
},ag.delay),(ag.delay<0||ag.combined)&&(K.e=H(ag.throttle,function(a){"resize"===a.type&&(G=U=-1),Q(a.all)
}),K.a=function(a){a=ae(a),Z.push.apply(Z,a)
},K.g=function(){return Z=j(Z).filter(function(){return !j(this).data(ag.loadedName)
})
},K.f=function(a){for(var l=0;
l<a.length;
l++){var i=Z.filter(function(){return this===a[l]
});
i.length&&Q(!1,i)
}},Q(),j(ag.appendScroll).on("scroll."+Y+" resize."+Y,K.e))
}function ae(D){var r=ag.defaultImage,m=ag.placeholder,C=ag.imageBase,p=ag.srcsetAttribute,y=ag.loaderAttribute,B=ag._f||{};
D=j(D).filter(function(){var i=j(this),l=X(this);
return !i.data(ag.handledName)&&(i.attr(ag.attribute)||i.attr(p)||i.attr(y)||B[l]!==g)
}).data("plugin_"+ag.name,S);
for(var E=0,z=D.length;
E<z;
E++){var a=j(D[E]),w=X(D[E]),v=a.attr(ag.imageBaseAttribute)||C;
w===n&&v&&a.attr(p)&&a.attr(p,af(a.attr(p),v)),B[w]===g||a.attr(y)||a.attr(y,B[w]),w===n&&r&&!a.attr(M)?a.attr(M,r):w===n||!m||a.css(k)&&"none"!==a.css(k)||a.css(k,"url('"+m+"')")
}return D
}function Q(E,z){if(!Z.length){return void (ag.autoDestroy&&S.destroy())
}for(var i=z||Z,D=!1,m=ag.imageBase||"",y=ag.srcsetAttribute,A=ag.handledName,F=0;
F<i.length;
F++){if(E||z||V(i[F])){var w=j(i[F]),r=X(i[F]),B=w.attr(ag.attribute),C=w.attr(ag.imageBaseAttribute)||m,a=w.attr(ag.loaderAttribute);
w.data(A)||ag.visibleOnly&&!w.is(":visible")||!((B||w.attr(y))&&(r===n&&(C+B!==w.attr(M)||w.attr(y)!==w.attr(J))||r!==n&&C+B!==w.css(k))||a)||(D=!0,w.data(A,!0),ad(w,r,C,a))
}}D&&(Z=j(Z).filter(function(){return !j(this).data(A)
}))
}function ad(N,E,a,z){++q;
var v=function(){t("onError",N),W(),v=j.noop
};
t("beforeLoad",N);
var L=ag.attribute,y=ag.srcsetAttribute,D=ag.sizesAttribute,I=ag.retinaAttribute,O=ag.removeAttribute,F=ag.loadedName,p=N.attr(I);
if(z){var C=function(){O&&N.removeAttr(ag.loaderAttribute),N.data(F,!0),t(e,N),setTimeout(W,1),C=j.noop
};
N.off(x).one(x,v).one(P,C),t(z,N,function(i){i?(N.off(P),C()):(N.off(x),v())
})||N.trigger(x)
}else{var B=j(new Image);
B.one(x,v).one(P,function(){N.hide(),E===n?N.attr(R,B.attr(R)).attr(J,B.attr(J)).attr(M,B.attr(M)):N.css(k,"url('"+B.attr(M)+"')"),N[ag.effect](ag.effectTime),O&&(N.removeAttr(L+" "+y+" "+I+" "+ag.imageBaseAttribute),D!==R&&N.removeAttr(D)),N.data(F,!0),t(e,N),B.remove(),W()
});
var w=(o&&p?p:N.attr(L))||"";
B.attr(R,N.attr(D)).attr(J,N.attr(y)).attr(M,w?a+w:null),B.complete&&B.trigger(P)
}}function V(l){var p=l.getBoundingClientRect(),m=ag.scrollDirection,u=ag.threshold,a=aa()+u>p.top&&-u<p.bottom,s=ab()+u>p.left&&-u<p.right;
return"vertical"===m?a:"horizontal"===m?s:a&&s
}function ab(){return G>=0?G:G=j(d).width()
}function aa(){return U>=0?U:U=j(d).height()
}function X(a){return a.tagName.toLowerCase()
}function af(l,p){if(p){var m=l.split(",");
l="";
for(var i=0,s=m.length;
i<s;
i++){l+=p+m[i].trim()+(i!==s-1?",":"")
}}return l
}function H(l,m){var p,a=0;
return function(v,r){function i(){a=+new Date,m.call(S,v)
}var s=+new Date-a;
p&&clearTimeout(p),s>l||!ag.enableThrottle||r?i():p=setTimeout(i,l-s)
}
}function W(){--q,Z.length||q||t("onFinishedAll")
}function t(a,i,l){return !!(a=ag[a])&&(a.apply(S,[].slice.call(arguments,1)),!0)
}var q=0,G=-1,U=-1,o=!1,e="afterLoad",P="load",x="error",n="img",M="src",J="srcset",R="sizes",k="background-image";
"event"===ag.bind||h?ac():j(d).on(P+"."+Y,ac)
}function b(i,n){var k=this,e=j.extend({},k.config,n),m={},p=e.name+"-"+ ++c;
return k.config=function(a,l){return l===g?e[a]:(e[a]=l,k)
},k.addItems=function(a){return m.a&&m.a("string"===j.type(a)?j(a):a),k
},k.getItems=function(){return m.g?m.g():{}
},k.update=function(a){return m.e&&m.e({},!a),k
},k.force=function(a){return m.f&&m.f("string"===j.type(a)?j(a):a),k
},k.loadAll=function(){return m.e&&m.e({all:!0},!0),k
},k.destroy=function(){return j(e.appendScroll).off("."+p,m.e),j(d).off("."+p),m={},g
},f(k,e,i,m,p),e.chainable?i:k
}var j=d.jQuery||d.Zepto,c=0,h=!1;
j.fn.Lazy=j.fn.lazy=function(a){return new b(this,a)
},j.Lazy=j.lazy=function(v,a,m){if(j.isFunction(a)&&(m=a,a=[]),j.isFunction(m)){v=j.isArray(v)?v:[v],a=j.isArray(a)?a:[a];
for(var e=b.prototype.config,q=e._f||(e._f={}),k=0,n=v.length;
k<n;
k++){(e[v[k]]===g||j.isFunction(e[v[k]]))&&(e[v[k]]=m)
}for(var p=0,w=a.length;
p<w;
p++){q[a[p]]=v[0]
}}},b.prototype.config={name:"lazy",chainable:!0,autoDestroy:!0,bind:"load",threshold:500,visibleOnly:!1,appendScroll:d,scrollDirection:"both",imageBase:null,defaultImage:"data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",placeholder:null,delay:-1,combined:!1,attribute:"data-src",srcsetAttribute:"data-srcset",sizesAttribute:"data-sizes",retinaAttribute:"data-retina",loaderAttribute:"data-loader",imageBaseAttribute:"data-imagebase",removeAttribute:!0,handledName:"handled",loadedName:"loaded",effect:"show",effectTime:0,enableThrottle:!0,throttle:250,beforeLoad:g,afterLoad:g,onError:g,onFinishedAll:g},j(d).on("load",function(){h=!0
})
}(window);
/*! jQuery & Zepto Lazy - All Plugins v1.7.7 - http://jquery.eisbehr.de/lazy - MIT&GPL-2.0 license - Copyright 2012-2017 Daniel 'Eisbehr' Kern */
;
!function(c){function b(d,g,f,h){c.ajax({url:g.attr("data-src"),type:h||"get",dataType:g.attr("data-type")||"html",success:function(a){g.html(a),f(!0),d.config("removeAttribute")&&g.removeAttr("data-src data-method data-type")
},error:function(){f(!1)
}})
}c.lazy("ajax",function(a,d){b(this,a,d,a.attr("data-method"))
}),c.lazy("get",function(a,d){b(this,a,d,"get")
}),c.lazy("post",function(a,d){b(this,a,d,"post")
})
}(window.jQuery||window.Zepto),function(a){a.lazy(["av","audio","video"],["audio","video"],function(b,h){var g=b[0].tagName.toLowerCase();
if("audio"===g||"video"===g){var j=b.find("data-src"),d=b.find("data-track"),l=0,k=function(){++l===j.length&&h(!1)
},f=function(){var c=a(this),m=c[0].tagName.toLowerCase(),i=c.prop("attributes"),n=a("data-src"===m?"<source>":"<track>");
"data-src"===m&&n.one("error",k),a.each(i,function(o,e){n.attr(e.name,e.value)
}),c.replaceWith(n)
};
b.one("loadedmetadata",function(){h(!0)
}).off("load error").attr("poster",b.attr("data-poster")),j.length?j.each(f):b.attr("data-src")?(a.each(b.attr("data-src").split(","),function(i,c){var m=c.split("|");
b.append(a("<source>").one("error",k).attr({src:m[0].trim(),type:m[1].trim()}))
}),this.config("removeAttribute")&&b.removeAttr("data-src")):h(!1),d.length&&d.each(f)
}else{h(!1)
}})
}(window.jQuery||window.Zepto),function(a){a.lazy(["frame","iframe"],"iframe",function(b,d){var c=this;
if("iframe"===b[0].tagName.toLowerCase()){var f=b.attr("data-error-detect");
"true"!==f&&"1"!==f?(b.attr("src",b.attr("data-src")),c.config("removeAttribute")&&b.removeAttr("data-src data-error-detect")):a.ajax({url:b.attr("data-src"),dataType:"html",crossDomain:!0,xhrFields:{withCredentials:!0},success:function(e){b.html(e).attr("src",b.attr("data-src")),c.config("removeAttribute")&&b.removeAttr("data-src data-error-detect")
},error:function(){d(!1)
}})
}else{d(!1)
}})
}(window.jQuery||window.Zepto),function(a){a.lazy("noop",function(){}),a.lazy("noop-success",function(c,b){b(!0)
}),a.lazy("noop-error",function(c,b){b(!1)
})
}(window.jQuery||window.Zepto),function(c){function b(h,k,j){var m=h.prop("attributes"),l=c("<"+k+">");
return c.each(m,function(i,e){"srcset"!==e.name&&e.name!==g||(e.value=d(e.value,j)),l.attr(e.name,e.value)
}),h.replaceWith(l),l
}function f(h,j,i){var k=c("<img>").one("load",function(){i(!0)
}).one("error",function(){i(!1)
}).appendTo(h).attr("src",j);
k.complete&&k.load()
}function d(i,h){if(h){var k=i.split(",");
i="";
for(var j=0,l=k.length;
j<l;
j++){i+=h+k[j].trim()+(j!==l-1?",":"")
}}return i
}var g="data-src";
c.lazy(["pic","picture"],["picture"],function(a,k){if("picture"===a[0].tagName.toLowerCase()){var j=a.find(g),e=a.find("data-img"),h=this.config("imageBase")||"";
j.length?(j.each(function(){b(c(this),"source",h)
}),1===e.length?(e=b(e,"img",h),e.on("load",function(){k(!0)
}).on("error",function(){k(!1)
}),e.attr("src",e.attr(g)),this.config("removeAttribute")&&e.removeAttr(g)):a.attr(g)?(f(a,h+a.attr(g),k),this.config("removeAttribute")&&a.removeAttr(g)):k(!1)):a.attr("data-srcset")?(c("<source>").attr({media:a.attr("data-media"),sizes:a.attr("data-sizes"),type:a.attr("data-type"),srcset:d(a.attr("data-srcset"),h)}).appendTo(a),f(a,h+a.attr(g),k),this.config("removeAttribute")&&a.removeAttr(g+" data-srcset data-media data-sizes data-type")):k(!1)
}else{k(!1)
}})
}(window.jQuery||window.Zepto),function(a){a.lazy(["js","javascript","script"],"script",function(c,b){"script"==c[0].tagName.toLowerCase()?(c.attr("src",c.attr("data-src")),this.config("removeAttribute")&&c.removeAttr("data-src")):b(!1)
})
}(window.jQuery||window.Zepto),function(a){a.lazy("vimeo",function(c,b){"iframe"===c[0].tagName.toLowerCase()?(c.attr("src","https://player.vimeo.com/video/"+c.attr("data-src")),this.config("removeAttribute")&&c.removeAttr("data-src")):b(!1)
})
}(window.jQuery||window.Zepto),function(a){a.lazy(["yt","youtube"],function(c,b){"iframe"===c[0].tagName.toLowerCase()?(c.attr("src","https://www.youtube.com/embed/"+c.attr("data-src")+"?rel=0&amp;showinfo=0"),this.config("removeAttribute")&&c.removeAttr("data-src")):b(!1)
})
}(window.jQuery||window.Zepto);
!function(a,b){"function"==typeof define&&define.amd?define(b):"object"==typeof exports?module.exports=b(require,exports,module):a.Tether=b()
}(this,function(aA,aP,aF){function aL(a,b){if(!(a instanceof b)){throw new TypeError("Cannot call a class as a function")
}}function aG(b){var f=b.getBoundingClientRect(),g={};
for(var a in f){g[a]=f[a]
}if(b.ownerDocument!==document){var d=b.ownerDocument.defaultView.frameElement;
if(d){var c=aG(d);
g.top+=c.top,g.bottom+=c.top,g.left+=c.left,g.right+=c.left
}}return g
}function aC(u){var p=getComputedStyle(u)||{},c=p.position,j=[];
if("fixed"===c){return[u]
}for(var d=u;
(d=d.parentNode)&&d&&1===d.nodeType;
){var b=void 0;
try{b=getComputedStyle(d)
}catch(v){}if("undefined"==typeof b||null===b){return j.push(d),j
}var q=b,m=q.overflow,g=q.overflowX,k=q.overflowY;
/(auto|scroll)/.test(m+k+g)&&("absolute"!==c||["relative","absolute","fixed"].indexOf(b.position)>=0)&&j.push(d)
}return j.push(u.ownerDocument.body),u.ownerDocument!==document&&j.push(u.ownerDocument.defaultView),j
}function aB(){ar&&document.body.removeChild(ar),ar=null
}function aT(b){var d=void 0;
b===document?(d=document,b=document.documentElement):d=b.ownerDocument;
var f=d.documentElement,a=aG(b),c=ae();
return a.top-=c.top,a.left-=c.left,"undefined"==typeof a.width&&(a.width=document.body.scrollWidth-a.left-a.right),"undefined"==typeof a.height&&(a.height=document.body.scrollHeight-a.top-a.bottom),a.top=a.top-f.clientTop,a.left=a.left-f.clientLeft,a.right=d.body.clientWidth-a.width-a.left,a.bottom=d.body.clientHeight-a.height-a.top,a
}function aO(a){return a.offsetParent||document.documentElement
}function aI(){if(ah){return ah
}var b=document.createElement("div");
b.style.width="100%",b.style.height="200px";
var c=document.createElement("div");
aM(c.style,{position:"absolute",top:0,left:0,pointerEvents:"none",visibility:"hidden",width:"200px",height:"150px",overflow:"hidden"}),c.appendChild(b),document.body.appendChild(c);
var d=b.offsetWidth;
c.style.overflow="scroll";
var a=b.offsetWidth;
d===a&&(a=c.clientWidth),document.body.removeChild(c);
var f=d-a;
return ah={width:f,height:f}
}function aM(){var a=arguments.length<=0||void 0===arguments[0]?{}:arguments[0],b=[];
return Array.prototype.push.apply(b,arguments),b.slice(1).forEach(function(c){if(c){for(var d in c){({}).hasOwnProperty.call(c,d)&&(a[d]=c[d])
}}}),a
}function aQ(b,c){if("undefined"!=typeof b.classList){c.split(" ").forEach(function(f){f.trim()&&b.classList.remove(f)
})
}else{var d=new RegExp("(^| )"+c.split(" ").join("|")+"( |$)","gi"),a=aR(b).replace(d," ");
aN(b,a)
}}function aE(a,b){if("undefined"!=typeof a.classList){b.split(" ").forEach(function(d){d.trim()&&a.classList.add(d)
})
}else{aQ(a,b);
var c=aR(a)+(" "+b);
aN(a,c)
}}function az(a,b){if("undefined"!=typeof a.classList){return a.classList.contains(b)
}var c=aR(a);
return new RegExp("(^| )"+b+"( |$)","gi").test(c)
}function aR(a){return a.className instanceof a.ownerDocument.defaultView.SVGAnimatedString?a.className.baseVal:a.className
}function aN(a,b){a.setAttribute("class",b)
}function aH(a,b,c){c.forEach(function(d){b.indexOf(d)===-1&&az(a,d)&&aQ(a,d)
}),b.forEach(function(d){az(a,d)||aE(a,d)
})
}function aL(a,b){if(!(a instanceof b)){throw new TypeError("Cannot call a class as a function")
}}function ay(a,b){if("function"!=typeof b&&null!==b){throw new TypeError("Super expression must either be null or a function, not "+typeof b)
}a.prototype=Object.create(b&&b.prototype,{constructor:{value:a,enumerable:!1,writable:!0,configurable:!0}}),b&&(Object.setPrototypeOf?Object.setPrototypeOf(a,b):a.__proto__=b)
}function av(a,b){var c=arguments.length<=2||void 0===arguments[2]?1:arguments[2];
return a+c>=b&&b>=a-c
}function aS(){return"undefined"!=typeof performance&&"undefined"!=typeof performance.now?performance.now():+new Date
}function ax(){for(var b={top:0,left:0},c=arguments.length,d=Array(c),a=0;
a<c;
a++){d[a]=arguments[a]
}return d.forEach(function(g){var h=g.top,f=g.left;
"string"==typeof h&&(h=parseFloat(h,10)),"string"==typeof f&&(f=parseFloat(f,10)),b.top+=h,b.left+=f
}),b
}function ap(a,b){return"string"==typeof a.left&&a.left.indexOf("%")!==-1&&(a.left=parseFloat(a.left,10)/100*b.width),"string"==typeof a.top&&a.top.indexOf("%")!==-1&&(a.top=parseFloat(a.top,10)/100*b.height),a
}function af(a,b){return"scrollParent"===b?b=a.scrollParents[0]:"window"===b&&(b=[pageXOffset,pageYOffset,innerWidth+pageXOffset,innerHeight+pageYOffset]),b===document&&(b=b.documentElement),"undefined"!=typeof b.nodeType&&!function(){var d=b,f=aT(b),c=f,g=getComputedStyle(b);
if(b=[c.left,c.top,f.width+c.left,f.height+c.top],d.ownerDocument!==document){var e=d.ownerDocument.defaultView;
b[0]+=e.pageXOffset,b[1]+=e.pageYOffset,b[2]+=e.pageXOffset,b[3]+=e.pageYOffset
}al.forEach(function(h,i){h=h[0].toUpperCase()+h.substr(1),"Top"===h||"Left"===h?b[i]+=parseFloat(g["border"+h+"Width"]):b[i]-=parseFloat(g["border"+h+"Width"])
})
}(),b
}var an=function(){function a(c,d){for(var f=0;
f<d.length;
f++){var b=d[f];
b.enumerable=b.enumerable||!1,b.configurable=!0,"value" in b&&(b.writable=!0),Object.defineProperty(c,b.key,b)
}}return function(c,d,b){return d&&a(c.prototype,d),b&&a(c,b),c
}
}(),aw=void 0;
"undefined"==typeof aw&&(aw={modules:[]});
var ar=null,ab=function(){var a=0;
return function(){return ++a
}
}(),ac={},ae=function(){var a=ar;
a&&document.body.contains(a)||(a=document.createElement("div"),a.setAttribute("data-tether-id",ab()),aM(a.style,{top:0,left:0,position:"absolute"}),document.body.appendChild(a),ar=a);
var b=a.getAttribute("data-tether-id");
return"undefined"==typeof ac[b]&&(ac[b]=aG(a),aJ(function(){delete ac[b]
})),ac[b]
},ah=null,Q=[],aJ=function(a){Q.push(a)
},aU=function(){for(var a=void 0;
a=Q.pop();
){a()
}},aq=function(){function a(){aL(this,a)
}return an(a,[{key:"on",value:function(c,d,f){var b=!(arguments.length<=3||void 0===arguments[3])&&arguments[3];
"undefined"==typeof this.bindings&&(this.bindings={}),"undefined"==typeof this.bindings[c]&&(this.bindings[c]=[]),this.bindings[c].push({handler:d,ctx:f,once:b})
}},{key:"once",value:function(b,c,d){this.on(b,c,d,!0)
}},{key:"off",value:function(b,c){if("undefined"!=typeof this.bindings&&"undefined"!=typeof this.bindings[b]){if("undefined"==typeof c){delete this.bindings[b]
}else{for(var d=0;
d<this.bindings[b].length;
){this.bindings[b][d].handler===c?this.bindings[b].splice(d,1):++d
}}}}},{key:"trigger",value:function(p){if("undefined"!=typeof this.bindings&&this.bindings[p]){for(var k=0,c=arguments.length,h=Array(c>1?c-1:0),d=1;
d<c;
d++){h[d-1]=arguments[d]
}for(;
k<this.bindings[p].length;
){var b=this.bindings[p][k],q=b.handler,m=b.ctx,j=b.once,g=m;
"undefined"==typeof g&&(g=this),q.apply(g,h),j?this.bindings[p].splice(k,1):++k
}}}}]),a
}();
aw.Utils={getActualBoundingClientRect:aG,getScrollParents:aC,getBounds:aT,getOffsetParent:aO,extend:aM,addClass:aE,removeClass:aQ,hasClass:az,updateClasses:aH,defer:aJ,flush:aU,uniqueId:ab,Evented:aq,getScrollBarSize:aI,removeUtilElements:aB};
var au=function(){function a(l,j){var c=[],g=!0,d=!1,b=void 0;
try{for(var m,k=l[Symbol.iterator]();
!(g=(m=k.next()).done)&&(c.push(m.value),!j||c.length!==j);
g=!0){}}catch(h){d=!0,b=h
}finally{try{!g&&k["return"]&&k["return"]()
}finally{if(d){throw b
}}}return c
}return function(b,c){if(Array.isArray(b)){return b
}if(Symbol.iterator in Object(b)){return a(b,c)
}throw new TypeError("Invalid attempt to destructure non-iterable instance")
}
}(),an=function(){function a(c,d){for(var f=0;
f<d.length;
f++){var b=d[f];
b.enumerable=b.enumerable||!1,b.configurable=!0,"value" in b&&(b.writable=!0),Object.defineProperty(c,b.key,b)
}}return function(c,d,b){return d&&a(c.prototype,d),b&&a(c,b),c
}
}(),aK=function(p,k,c){for(var h=!0;
h;
){var d=p,b=k,q=c;
h=!1,null===d&&(d=Function.prototype);
var m=Object.getOwnPropertyDescriptor(d,b);
if(void 0!==m){if("value" in m){return m.value
}var j=m.get;
if(void 0===j){return
}return j.call(q)
}var g=Object.getPrototypeOf(d);
if(null===g){return
}p=g,k=b,c=q,h=!0,m=g=void 0
}};
if("undefined"==typeof aw){throw new Error("You must include the utils.js file before tether.js")
}var J=aw.Utils,aC=J.getScrollParents,aT=J.getBounds,aO=J.getOffsetParent,aM=J.extend,aE=J.addClass,aQ=J.removeClass,aH=J.updateClasses,aJ=J.defer,aU=J.flush,aI=J.getScrollBarSize,aB=J.removeUtilElements,ai=function(){if("undefined"==typeof document){return""
}for(var b=document.createElement("div"),c=["transform","WebkitTransform","OTransform","MozTransform","msTransform"],d=0;
d<c.length;
++d){var a=c[d];
if(void 0!==b.style[a]){return a
}}}(),ao=[],K=function(){ao.forEach(function(a){a.position(!1)
}),aU()
};
!function(){var b=null,c=null,d=null,a=function f(){return"undefined"!=typeof c&&c>16?(c=Math.min(c-16,250),void (d=setTimeout(f,250))):void ("undefined"!=typeof b&&aS()-b<10||(null!=d&&(clearTimeout(d),d=null),b=aS(),K(),c=aS()-b))
};
"undefined"!=typeof window&&"undefined"!=typeof window.addEventListener&&["resize","scroll","touchmove"].forEach(function(e){window.addEventListener(e,a)
})
}();
var am={center:"center",left:"right",right:"left"},ak={middle:"middle",top:"bottom",bottom:"top"},ag={top:0,left:0,middle:"50%",center:"50%",bottom:"100%",right:"100%"},aa=function(b,c){var d=b.left,a=b.top;
return"auto"===d&&(d=am[c.left]),"auto"===a&&(a=ak[c.top]),{left:d,top:a}
},Z=function(a){var b=a.left,c=a.top;
return"undefined"!=typeof ag[a.left]&&(b=ag[a.left]),"undefined"!=typeof ag[a.top]&&(c=ag[a.top]),{left:b,top:c}
},ad=function(b){var c=b.split(" "),d=au(c,2),a=d[0],f=d[1];
return{top:a,left:f}
},aD=ad,aj=function(a){function b(c){var d=this;
aL(this,b),aK(Object.getPrototypeOf(b.prototype),"constructor",this).call(this),this.position=this.position.bind(this),ao.push(this),this.history=[],this.setOptions(c,!1),aw.modules.forEach(function(e){"undefined"!=typeof e.initialize&&e.initialize.call(d)
}),this.position()
}return ay(b,a),an(b,[{key:"getClass",value:function(){var c=arguments.length<=0||void 0===arguments[0]?"":arguments[0],d=this.options.classes;
return"undefined"!=typeof d&&d[c]?this.options.classes[c]:this.options.classPrefix?this.options.classPrefix+"-"+c:c
}},{key:"setOptions",value:function(g){var k=this,l=arguments.length<=1||void 0===arguments[1]||arguments[1],d={offset:"0 0",targetOffset:"0 0",targetAttachment:"auto auto",classPrefix:"tether"};
this.options=aM(d,g);
var m=this.options,h=m.element,c=m.target,j=m.targetModifier;
if(this.element=h,this.target=c,this.targetModifier=j,"viewport"===this.target?(this.target=document.body,this.targetModifier="visible"):"scroll-handle"===this.target&&(this.target=document.body,this.targetModifier="scroll-handle"),["element","target"].forEach(function(e){if("undefined"==typeof k[e]){throw new Error("Tether Error: Both element and target must be defined")
}"undefined"!=typeof k[e].jquery?k[e]=k[e][0]:"string"==typeof k[e]&&(k[e]=document.querySelector(k[e]))
}),aE(this.element,this.getClass("element")),this.options.addTargetClasses!==!1&&aE(this.target,this.getClass("target")),!this.options.attachment){throw new Error("Tether Error: You must provide an attachment")
}this.targetAttachment=aD(this.options.targetAttachment),this.attachment=aD(this.options.attachment),this.offset=ad(this.options.offset),this.targetOffset=ad(this.options.targetOffset),"undefined"!=typeof this.scrollParents&&this.disable(),"scroll-handle"===this.targetModifier?this.scrollParents=[this.target]:this.scrollParents=aC(this.target),this.options.enabled!==!1&&this.enable(l)
}},{key:"getTargetBounds",value:function(){if("undefined"==typeof this.targetModifier){return aT(this.target)
}if("visible"===this.targetModifier){if(this.target===document.body){return{top:pageYOffset,left:pageXOffset,height:innerHeight,width:innerWidth}
}var p=aT(this.target),m={height:p.height,width:p.width,top:p.top,left:p.left};
return m.height=Math.min(m.height,p.height-(pageYOffset-p.top)),m.height=Math.min(m.height,p.height-(p.top+p.height-(pageYOffset+innerHeight))),m.height=Math.min(innerHeight,m.height),m.height-=2,m.width=Math.min(m.width,p.width-(pageXOffset-p.left)),m.width=Math.min(m.width,p.width-(p.left+p.width-(pageXOffset+innerWidth))),m.width=Math.min(innerWidth,m.width),m.width-=2,m.top<pageYOffset&&(m.top=pageYOffset),m.left<pageXOffset&&(m.left=pageXOffset),m
}if("scroll-handle"===this.targetModifier){var p=void 0,d=this.target;
d===document.body?(d=document.documentElement,p={left:pageXOffset,top:pageYOffset,height:innerHeight,width:innerWidth}):p=aT(d);
var j=getComputedStyle(d),g=d.scrollWidth>d.clientWidth||[j.overflow,j.overflowX].indexOf("scroll")>=0||this.target!==document.body,c=0;
g&&(c=15);
var q=p.height-parseFloat(j.borderTopWidth)-parseFloat(j.borderBottomWidth)-c,m={width:15,height:0.975*q*(q/d.scrollHeight),left:p.left+p.width-parseFloat(j.borderLeftWidth)-15},k=0;
q<408&&this.target===document.body&&(k=-0.00011*Math.pow(q,2)-0.00727*q+22.58),this.target!==document.body&&(m.height=Math.max(m.height,24));
var h=this.target.scrollTop/(d.scrollHeight-q);
return m.top=h*(q-m.height-k)+p.top+parseFloat(j.borderTopWidth),this.target===document.body&&(m.height=Math.max(m.height,24)),m
}}},{key:"clearCache",value:function(){this._cache={}
}},{key:"cache",value:function(c,d){return"undefined"==typeof this._cache&&(this._cache={}),"undefined"==typeof this._cache[c]&&(this._cache[c]=d.call(this)),this._cache[c]
}},{key:"enable",value:function(){var c=this,d=arguments.length<=0||void 0===arguments[0]||arguments[0];
this.options.addTargetClasses!==!1&&aE(this.target,this.getClass("enabled")),aE(this.element,this.getClass("enabled")),this.enabled=!0,this.scrollParents.forEach(function(f){f!==c.target.ownerDocument&&f.addEventListener("scroll",c.position)
}),d&&this.position()
}},{key:"disable",value:function(){var c=this;
aQ(this.target,this.getClass("enabled")),aQ(this.element,this.getClass("enabled")),this.enabled=!1,"undefined"!=typeof this.scrollParents&&this.scrollParents.forEach(function(d){d.removeEventListener("scroll",c.position)
})
}},{key:"destroy",value:function(){var c=this;
this.disable(),ao.forEach(function(d,f){d===c&&ao.splice(f,1)
}),0===ao.length&&aB()
}},{key:"updateAttachClasses",value:function(d,g){var h=this;
d=d||this.attachment,g=g||this.targetAttachment;
var c=["left","top","bottom","right","middle","center"];
"undefined"!=typeof this._addAttachClasses&&this._addAttachClasses.length&&this._addAttachClasses.splice(0,this._addAttachClasses.length),"undefined"==typeof this._addAttachClasses&&(this._addAttachClasses=[]);
var j=this._addAttachClasses;
d.top&&j.push(this.getClass("element-attached")+"-"+d.top),d.left&&j.push(this.getClass("element-attached")+"-"+d.left),g.top&&j.push(this.getClass("target-attached")+"-"+g.top),g.left&&j.push(this.getClass("target-attached")+"-"+g.left);
var f=[];
c.forEach(function(e){f.push(h.getClass("element-attached")+"-"+e),f.push(h.getClass("target-attached")+"-"+e)
}),aJ(function(){"undefined"!=typeof h._addAttachClasses&&(aH(h.element,h._addAttachClasses,f),h.options.addTargetClasses!==!1&&aH(h.target,h._addAttachClasses,f),delete h._addAttachClasses)
})
}},{key:"position",value:function(){var z=this,P=arguments.length<=0||void 0===arguments[0]||arguments[0];
if(this.enabled){this.clearCache();
var G=aa(this.targetAttachment,this.attachment);
this.updateAttachClasses(this.attachment,G);
var L=this.cache("element-bounds",function(){return aT(z.element)
}),H=L.width,C=L.height;
if(0===H&&0===C&&"undefined"!=typeof this.lastSize){var B=this.lastSize;
H=B.width,C=B.height
}else{this.lastSize={width:H,height:C}
}var M=this.cache("target-bounds",function(){return z.getTargetBounds()
}),R=M,F=ap(Z(this.attachment),{width:H,height:C}),w=ap(Z(G),R),U=ap(this.offset,{width:H,height:C}),N=ap(this.targetOffset,R);
F=ax(F,U),w=ax(w,N);
for(var I=M.left+w.left-F.left,q=M.top+w.top-F.top,l=0;
l<aw.modules.length;
++l){var V=aw.modules[l],k=V.position.call(this,{left:I,top:q,targetAttachment:G,targetPos:M,elementPos:L,offset:F,targetOffset:w,manualOffset:U,manualTargetOffset:N,scrollbarSize:j,attachment:this.attachment});
if(k===!1){return !1
}"undefined"!=typeof k&&"object"==typeof k&&(q=k.top,I=k.left)
}var x={page:{top:q,left:I},viewport:{top:q-pageYOffset,bottom:pageYOffset-q-C+innerHeight,left:I-pageXOffset,right:pageXOffset-I-H+innerWidth}},D=this.target.ownerDocument,f=D.defaultView,j=void 0;
return f.innerHeight>D.documentElement.clientHeight&&(j=this.cache("scrollbar-size",aI),x.viewport.bottom-=j.height),f.innerWidth>D.documentElement.clientWidth&&(j=this.cache("scrollbar-size",aI),x.viewport.right-=j.width),["","static"].indexOf(D.body.style.position)!==-1&&["","static"].indexOf(D.body.parentElement.style.position)!==-1||(x.page.bottom=D.body.scrollHeight-q-C,x.page.right=D.body.scrollWidth-I-H),"undefined"!=typeof this.options.optimizations&&this.options.optimizations.moveElement!==!1&&"undefined"==typeof this.targetModifier&&!function(){var m=z.cache("target-offsetparent",function(){return aO(z.target)
}),p=z.cache("target-offsetparent-bounds",function(){return aT(m)
}),d=getComputedStyle(m),t=p,h={};
if(["Top","Left","Bottom","Right"].forEach(function(e){h[e.toLowerCase()]=parseFloat(d["border"+e+"Width"])
}),p.right=D.body.scrollWidth-p.left-t.width+h.right,p.bottom=D.body.scrollHeight-p.top-t.height+h.bottom,x.page.top>=p.top+h.top&&x.page.bottom>=p.bottom&&x.page.left>=p.left+h.left&&x.page.right>=p.right){var g=m.scrollTop,c=m.scrollLeft;
x.offset={top:x.page.top-p.top+g-h.top,left:x.page.left-p.left+c-h.left}
}}(),this.move(x),this.history.unshift(x),this.history.length>3&&this.history.pop(),P&&aU(),!0
}}},{key:"move",value:function(G){var z=this;
if("undefined"!=typeof this.element.parentNode){var j={};
for(var x in G){j[x]={};
for(var k in G[x]){for(var f=!1,H=0;
H<this.history.length;
++H){var D=this.history[H];
if("undefined"!=typeof D[x]&&!av(D[x][k],G[x][k])){f=!0;
break
}}f||(j[x][k]=!0)
}}var w={top:"",left:"",right:"",bottom:""},A=function(e,m){var d="undefined"!=typeof z.options.optimizations,p=d?z.options.optimizations.gpu:null;
if(p!==!1){var l=void 0,g=void 0;
if(e.top?(w.top=0,l=m.top):(w.bottom=0,l=-m.bottom),e.left?(w.left=0,g=m.left):(w.right=0,g=-m.right),window.matchMedia){var c=window.matchMedia("only screen and (min-resolution: 1.3dppx)").matches||window.matchMedia("only screen and (-webkit-min-device-pixel-ratio: 1.3)").matches;
c||(g=Math.round(g),l=Math.round(l))
}w[ai]="translateX("+g+"px) translateY("+l+"px)","msTransform"!==ai&&(w[ai]+=" translateZ(0)")
}else{e.top?w.top=m.top+"px":w.bottom=m.bottom+"px",e.left?w.left=m.left+"px":w.right=m.right+"px"
}},h=!1;
if((j.page.top||j.page.bottom)&&(j.page.left||j.page.right)?(w.position="absolute",A(j.page,G.page)):(j.viewport.top||j.viewport.bottom)&&(j.viewport.left||j.viewport.right)?(w.position="fixed",A(j.viewport,G.viewport)):"undefined"!=typeof j.offset&&j.offset.top&&j.offset.left?!function(){w.position="absolute";
var c=z.cache("target-offsetparent",function(){return aO(z.target)
});
aO(z.element)!==c&&aJ(function(){z.element.parentNode.removeChild(z.element),c.appendChild(z.element)
}),A(j.offset,G.offset),h=!0
}():(w.position="absolute",A({top:!0,left:!0},G.page)),!h){if(this.options.bodyElement){this.options.bodyElement.appendChild(this.element)
}else{for(var F=!0,B=this.element.parentNode;
B&&1===B.nodeType&&"BODY"!==B.tagName;
){if("static"!==getComputedStyle(B).position){F=!1;
break
}B=B.parentNode
}F||(this.element.parentNode.removeChild(this.element),this.element.ownerDocument.body.appendChild(this.element))
}}var y={},q=!1;
for(var k in w){var E=w[k],C=this.element.style[k];
C!==E&&(q=!0,y[k]=E)
}q&&aJ(function(){aM(z.element.style,y),z.trigger("repositioned")
})
}}}]),b
}(aq);
aj.modules=[],aw.position=K;
var at=aM(aj,aw),au=function(){function a(l,j){var c=[],g=!0,d=!1,b=void 0;
try{for(var m,k=l[Symbol.iterator]();
!(g=(m=k.next()).done)&&(c.push(m.value),!j||c.length!==j);
g=!0){}}catch(h){d=!0,b=h
}finally{try{!g&&k["return"]&&k["return"]()
}finally{if(d){throw b
}}}return c
}return function(b,c){if(Array.isArray(b)){return b
}if(Symbol.iterator in Object(b)){return a(b,c)
}throw new TypeError("Invalid attempt to destructure non-iterable instance")
}
}(),J=aw.Utils,aT=J.getBounds,aM=J.extend,aH=J.updateClasses,aJ=J.defer,al=["left","top","right","bottom"];
aw.modules.push({position:function(E){var x=this,h=E.top,m=E.left,j=E.targetAttachment;
if(!this.options.constraints){return !0
}var a=this.cache("element-bounds",function(){return aT(x.element)
}),F=a.height,w=a.width;
if(0===w&&0===F&&"undefined"!=typeof this.lastSize){var k=this.lastSize;
w=k.width,F=k.height
}var z=this.cache("target-bounds",function(){return x.getTargetBounds()
}),b=z.height,D=z.width,A=[this.getClass("pinned"),this.getClass("out-of-bounds")];
this.options.constraints.forEach(function(c){var d=c.outOfBoundsClass,f=c.pinnedClass;
d&&A.push(d),f&&A.push(f)
}),A.forEach(function(c){["left","top","right","bottom"].forEach(function(d){A.push(c+"-"+d)
})
});
var q=[],C=aM({},j),B=aM({},this.attachment);
return this.options.constraints.forEach(function(y){var e=y.to,u=y.attachment,i=y.pin;
"undefined"==typeof u&&(u="");
var n=void 0,o=void 0;
if(u.indexOf(" ")>=0){var p=u.split(" "),g=au(p,2);
o=g[0],n=g[1]
}else{n=o=u
}var s=af(x,e);
"target"!==o&&"both"!==o||(h<s[1]&&"top"===C.top&&(h+=b,C.top="bottom"),h+F>s[3]&&"bottom"===C.top&&(h-=b,C.top="top")),"together"===o&&("top"===C.top&&("bottom"===B.top&&h<s[1]?(h+=b,C.top="bottom",h+=F,B.top="top"):"top"===B.top&&h+F>s[3]&&h-(F-b)>=s[1]&&(h-=F-b,C.top="bottom",B.top="bottom")),"bottom"===C.top&&("top"===B.top&&h+F>s[3]?(h-=b,C.top="top",h-=F,B.top="bottom"):"bottom"===B.top&&h<s[1]&&h+(2*F-b)<=s[3]&&(h+=F-b,C.top="top",B.top="top")),"middle"===C.top&&(h+F>s[3]&&"top"===B.top?(h-=F,B.top="bottom"):h<s[1]&&"bottom"===B.top&&(h+=F,B.top="top"))),"target"!==n&&"both"!==n||(m<s[0]&&"left"===C.left&&(m+=D,C.left="right"),m+w>s[2]&&"right"===C.left&&(m-=D,C.left="left")),"together"===n&&(m<s[0]&&"left"===C.left?"right"===B.left?(m+=D,C.left="right",m+=w,B.left="left"):"left"===B.left&&(m+=D,C.left="right",m-=w,B.left="right"):m+w>s[2]&&"right"===C.left?"left"===B.left?(m-=D,C.left="left",m-=w,B.left="right"):"right"===B.left&&(m-=D,C.left="left",m+=w,B.left="left"):"center"===C.left&&(m+w>s[2]&&"left"===B.left?(m-=w,B.left="right"):m<s[0]&&"right"===B.left&&(m+=w,B.left="left"))),"element"!==o&&"both"!==o||(h<s[1]&&"bottom"===B.top&&(h+=F,B.top="top"),h+F>s[3]&&"top"===B.top&&(h-=F,B.top="bottom")),"element"!==n&&"both"!==n||(m<s[0]&&("right"===B.left?(m+=w,B.left="left"):"center"===B.left&&(m+=w/2,B.left="left")),m+w>s[2]&&("left"===B.left?(m-=w,B.left="right"):"center"===B.left&&(m-=w/2,B.left="right"))),"string"==typeof i?i=i.split(",").map(function(c){return c.trim()
}):i===!0&&(i=["top","left","right","bottom"]),i=i||[];
var v=[],f=[];
h<s[1]&&(i.indexOf("top")>=0?(h=s[1],v.push("top")):f.push("top")),h+F>s[3]&&(i.indexOf("bottom")>=0?(h=s[3]-F,v.push("bottom")):f.push("bottom")),m<s[0]&&(i.indexOf("left")>=0?(m=s[0],v.push("left")):f.push("left")),m+w>s[2]&&(i.indexOf("right")>=0?(m=s[2]-w,v.push("right")):f.push("right")),v.length&&!function(){var c=void 0;
c="undefined"!=typeof x.options.pinnedClass?x.options.pinnedClass:x.getClass("pinned"),q.push(c),v.forEach(function(d){q.push(c+"-"+d)
})
}(),f.length&&!function(){var c=void 0;
c="undefined"!=typeof x.options.outOfBoundsClass?x.options.outOfBoundsClass:x.getClass("out-of-bounds"),q.push(c),f.forEach(function(d){q.push(c+"-"+d)
})
}(),(v.indexOf("left")>=0||v.indexOf("right")>=0)&&(B.left=C.left=!1),(v.indexOf("top")>=0||v.indexOf("bottom")>=0)&&(B.top=C.top=!1),C.top===j.top&&C.left===j.left&&B.top===x.attachment.top&&B.left===x.attachment.left||(x.updateAttachClasses(B,C),x.trigger("update",{attachment:B,targetAttachment:C}))
}),aJ(function(){x.options.addTargetClasses!==!1&&aH(x.target,q,A),aH(x.element,q,A)
}),{top:h,left:m}
}});
var J=aw.Utils,aT=J.getBounds,aH=J.updateClasses,aJ=J.defer;
aw.modules.push({position:function(A){var w=this,g=A.top,m=A.left,j=this.cache("element-bounds",function(){return aT(w.element)
}),a=j.height,B=j.width,v=this.getTargetBounds(),k=g+a,q=m+B,x=[];
g<=v.bottom&&k>=v.top&&["left","right"].forEach(function(c){var d=v[c];
d!==m&&d!==q||x.push(c)
}),m<=v.right&&q>=v.left&&["top","bottom"].forEach(function(c){var d=v[c];
d!==g&&d!==k||x.push(c)
});
var b=[],z=[],y=["left","top","right","bottom"];
return b.push(this.getClass("abutted")),y.forEach(function(c){b.push(w.getClass("abutted")+"-"+c)
}),x.length&&z.push(this.getClass("abutted")),x.forEach(function(c){z.push(w.getClass("abutted")+"-"+c)
}),aJ(function(){w.options.addTargetClasses!==!1&&aH(w.target,z,b),aH(w.element,z,b)
}),!0
}});
var au=function(){function a(l,j){var c=[],g=!0,d=!1,b=void 0;
try{for(var m,k=l[Symbol.iterator]();
!(g=(m=k.next()).done)&&(c.push(m.value),!j||c.length!==j);
g=!0){}}catch(h){d=!0,b=h
}finally{try{!g&&k["return"]&&k["return"]()
}finally{if(d){throw b
}}}return c
}return function(b,c){if(Array.isArray(b)){return b
}if(Symbol.iterator in Object(b)){return a(b,c)
}throw new TypeError("Invalid attempt to destructure non-iterable instance")
}
}();
return aw.modules.push({position:function(d){var h=d.top,j=d.left;
if(this.options.shift){var c=this.options.shift;
"function"==typeof this.options.shift&&(c=this.options.shift.call(this,{top:h,left:j}));
var k=void 0,g=void 0;
if("string"==typeof c){c=c.split(" "),c[1]=c[1]||c[0];
var f=c,b=au(f,2);
k=b[0],g=b[1],k=parseFloat(k,10),g=parseFloat(g,10)
}else{k=c.top,g=c.left
}return h+=k,j+=g,{top:h,left:j}
}}}),at
});
(function(a){if(typeof define==="function"&&define.amd){define(["jquery"],a)
}else{if(typeof exports!=="undefined"){module.exports=a(require("jquery"))
}else{a(jQuery)
}}}(function(a){var b=window.Slick||{};
b=(function(){var c=0;
function d(g,h){var f=this,e;
f.defaults={accessibility:true,adaptiveHeight:false,appendArrows:a(g),appendDots:a(g),arrows:true,asNavFor:null,prevArrow:'<button type="button" data-role="none" class="slick-prev" aria-label="Previous" tabindex="0" role="button">Previous</button>',nextArrow:'<button type="button" data-role="none" class="slick-next" aria-label="Next" tabindex="0" role="button">Next</button>',autoplay:false,autoplaySpeed:3000,centerMode:false,centerPadding:"50px",cssEase:"ease",customPaging:function(k,j){return a('<button type="button" data-role="none" role="button" tabindex="0" />').text(j+1)
},dots:false,dotsClass:"slick-dots",draggable:true,easing:"linear",edgeFriction:0.35,fade:false,focusOnSelect:false,infinite:true,initialSlide:0,lazyLoad:"ondemand",mobileFirst:false,pauseOnHover:true,pauseOnFocus:true,pauseOnDotsHover:false,respondTo:"window",responsive:null,rows:1,rtl:false,slide:"",slidesPerRow:1,slidesToShow:1,slidesToScroll:1,speed:500,swipe:true,swipeToSlide:false,touchMove:true,touchThreshold:5,useCSS:true,useTransform:true,variableWidth:false,vertical:false,verticalSwiping:false,waitForAnimate:true,zIndex:1000};
f.initials={animating:false,dragging:false,autoPlayTimer:null,currentDirection:0,currentLeft:null,currentSlide:0,direction:1,$dots:null,listWidth:null,listHeight:null,loadIndex:0,$nextArrow:null,$prevArrow:null,slideCount:null,slideWidth:null,$slideTrack:null,$slides:null,sliding:false,slideOffset:0,swipeLeft:null,$list:null,touchObject:{},transformsEnabled:false,unslicked:false};
a.extend(f,f.initials);
f.activeBreakpoint=null;
f.animType=null;
f.animProp=null;
f.breakpoints=[];
f.breakpointSettings=[];
f.cssTransitions=false;
f.focussed=false;
f.interrupted=false;
f.hidden="hidden";
f.paused=true;
f.positionProp=null;
f.respondTo=null;
f.rowCount=1;
f.shouldClick=true;
f.$slider=a(g);
f.$slidesCache=null;
f.transformType=null;
f.transitionType=null;
f.visibilityChange="visibilitychange";
f.windowWidth=0;
f.windowTimer=null;
e=a(g).data("slick")||{};
f.options=a.extend({},f.defaults,h,e);
f.currentSlide=f.options.initialSlide;
f.originalSettings=f.options;
if(typeof document.mozHidden!=="undefined"){f.hidden="mozHidden";
f.visibilityChange="mozvisibilitychange"
}else{if(typeof document.webkitHidden!=="undefined"){f.hidden="webkitHidden";
f.visibilityChange="webkitvisibilitychange"
}}f.autoPlay=a.proxy(f.autoPlay,f);
f.autoPlayClear=a.proxy(f.autoPlayClear,f);
f.autoPlayIterator=a.proxy(f.autoPlayIterator,f);
f.changeSlide=a.proxy(f.changeSlide,f);
f.clickHandler=a.proxy(f.clickHandler,f);
f.selectHandler=a.proxy(f.selectHandler,f);
f.setPosition=a.proxy(f.setPosition,f);
f.swipeHandler=a.proxy(f.swipeHandler,f);
f.dragHandler=a.proxy(f.dragHandler,f);
f.keyHandler=a.proxy(f.keyHandler,f);
f.instanceUid=c++;
f.htmlExpr=/^(?:\s*(<[\w\W]+>)[^>]*)$/;
f.registerBreakpoints();
f.init(true)
}return d
}());
b.prototype.activateADA=function(){var c=this;
c.$slideTrack.find(".slick-active").attr({"aria-hidden":"false"}).find("a, input, button, select").attr({tabindex:"0"})
};
b.prototype.addSlide=b.prototype.slickAdd=function(c,e,f){var d=this;
if(typeof(e)==="boolean"){f=e;
e=null
}else{if(e<0||(e>=d.slideCount)){return false
}}d.unload();
if(typeof(e)==="number"){if(e===0&&d.$slides.length===0){a(c).appendTo(d.$slideTrack)
}else{if(f){a(c).insertBefore(d.$slides.eq(e))
}else{a(c).insertAfter(d.$slides.eq(e))
}}}else{if(f===true){a(c).prependTo(d.$slideTrack)
}else{a(c).appendTo(d.$slideTrack)
}}d.$slides=d.$slideTrack.children(this.options.slide);
d.$slideTrack.children(this.options.slide).detach();
d.$slideTrack.append(d.$slides);
d.$slides.each(function(g,h){a(h).attr("data-slick-index",g)
});
d.$slidesCache=d.$slides;
d.reinit()
};
b.prototype.animateHeight=function(){var d=this;
if(d.options.slidesToShow===1&&d.options.adaptiveHeight===true&&d.options.vertical===false){var c=d.$slides.eq(d.currentSlide).outerHeight(true);
d.$list.animate({height:c},d.options.speed)
}};
b.prototype.animateSlide=function(f,e){var d={},c=this;
c.animateHeight();
if(c.options.rtl===true&&c.options.vertical===false){f=-f
}if(c.transformsEnabled===false){if(c.options.vertical===false){c.$slideTrack.animate({left:f},c.options.speed,c.options.easing,e)
}else{c.$slideTrack.animate({top:f},c.options.speed,c.options.easing,e)
}}else{if(c.cssTransitions===false){if(c.options.rtl===true){c.currentLeft=-(c.currentLeft)
}a({animStart:c.currentLeft}).animate({animStart:f},{duration:c.options.speed,easing:c.options.easing,step:function(g){g=Math.ceil(g);
if(c.options.vertical===false){d[c.animType]="translate("+g+"px, 0px)";
c.$slideTrack.css(d)
}else{d[c.animType]="translate(0px,"+g+"px)";
c.$slideTrack.css(d)
}},complete:function(){if(e){e.call()
}}})
}else{c.applyTransition();
f=Math.ceil(f);
if(c.options.vertical===false){d[c.animType]="translate3d("+f+"px, 0px, 0px)"
}else{d[c.animType]="translate3d(0px,"+f+"px, 0px)"
}c.$slideTrack.css(d);
if(e){setTimeout(function(){c.disableTransition();
e.call()
},c.options.speed)
}}}};
b.prototype.getNavTarget=function(){var d=this,c=d.options.asNavFor;
if(c&&c!==null){c=a(c).not(d.$slider)
}return c
};
b.prototype.asNavFor=function(e){var d=this,c=d.getNavTarget();
if(c!==null&&typeof c==="object"){c.each(function(){var f=a(this).slick("getSlick");
if(!f.unslicked){f.slideHandler(e,true)
}})
}};
b.prototype.applyTransition=function(c){var d=this,e={};
if(d.options.fade===false){e[d.transitionType]=d.transformType+" "+d.options.speed+"ms "+d.options.cssEase
}else{e[d.transitionType]="opacity "+d.options.speed+"ms "+d.options.cssEase
}if(d.options.fade===false){d.$slideTrack.css(e)
}else{d.$slides.eq(c).css(e)
}};
b.prototype.autoPlay=function(){var c=this;
c.autoPlayClear();
if(c.slideCount>c.options.slidesToShow){c.autoPlayTimer=setInterval(c.autoPlayIterator,c.options.autoplaySpeed)
}};
b.prototype.autoPlayClear=function(){var c=this;
if(c.autoPlayTimer){clearInterval(c.autoPlayTimer)
}};
b.prototype.autoPlayIterator=function(){var c=this,d=c.currentSlide+c.options.slidesToScroll;
if(!c.paused&&!c.interrupted&&!c.focussed){if(c.options.infinite===false){if(c.direction===1&&(c.currentSlide+1)===(c.slideCount-1)){c.direction=0
}else{if(c.direction===0){d=c.currentSlide-c.options.slidesToScroll;
if(c.currentSlide-1===0){c.direction=1
}}}}c.slideHandler(d)
}};
b.prototype.buildArrows=function(){var c=this;
if(c.options.arrows===true){c.$prevArrow=a(c.options.prevArrow).addClass("slick-arrow");
c.$nextArrow=a(c.options.nextArrow).addClass("slick-arrow");
if(c.slideCount>c.options.slidesToShow){c.$prevArrow.removeClass("slick-hidden").removeAttr("aria-hidden tabindex");
c.$nextArrow.removeClass("slick-hidden").removeAttr("aria-hidden tabindex");
if(c.htmlExpr.test(c.options.prevArrow)){c.$prevArrow.prependTo(c.options.appendArrows)
}if(c.htmlExpr.test(c.options.nextArrow)){c.$nextArrow.appendTo(c.options.appendArrows)
}if(c.options.infinite!==true){c.$prevArrow.addClass("slick-disabled").attr("aria-disabled","true")
}}else{c.$prevArrow.add(c.$nextArrow).addClass("slick-hidden").attr({"aria-disabled":"true",tabindex:"-1"})
}}};
b.prototype.buildDots=function(){var d=this,e,c;
if(d.options.dots===true&&d.slideCount>d.options.slidesToShow){d.$slider.addClass("slick-dotted");
c=a("<ul />").addClass(d.options.dotsClass);
for(e=0;
e<=d.getDotCount();
e+=1){c.append(a("<li />").append(d.options.customPaging.call(this,d,e)))
}d.$dots=c.appendTo(d.options.appendDots);
d.$dots.find("li").first().addClass("slick-active").attr("aria-hidden","false")
}};
b.prototype.buildOut=function(){var c=this;
c.$slides=c.$slider.children(c.options.slide+":not(.slick-cloned)").addClass("slick-slide");
c.slideCount=c.$slides.length;
c.$slides.each(function(d,e){a(e).attr("data-slick-index",d).data("originalStyling",a(e).attr("style")||"")
});
c.$slider.addClass("slick-slider");
c.$slideTrack=(c.slideCount===0)?a('<div class="slick-track"/>').appendTo(c.$slider):c.$slides.wrapAll('<div class="slick-track"/>').parent();
c.$list=c.$slideTrack.wrap('<div aria-live="polite" class="slick-list"/>').parent();
c.$slideTrack.css("opacity",0);
if(c.options.centerMode===true||c.options.swipeToSlide===true){c.options.slidesToScroll=1
}a("img[data-lazy]",c.$slider).not("[src]").addClass("slick-loading");
c.setupInfinite();
c.buildArrows();
c.buildDots();
c.updateDots();
c.setSlideClasses(typeof c.currentSlide==="number"?c.currentSlide:0);
if(c.options.draggable===true){c.$list.addClass("draggable")
}};
b.prototype.buildRows=function(){var m=this,l,k,i,d,j,h,e;
d=document.createDocumentFragment();
h=m.$slider.children();
if(m.options.rows>1){e=m.options.slidesPerRow*m.options.rows;
j=Math.ceil(h.length/e);
for(l=0;
l<j;
l++){var f=document.createElement("div");
for(k=0;
k<m.options.rows;
k++){var n=document.createElement("div");
for(i=0;
i<m.options.slidesPerRow;
i++){var g=(l*e+((k*m.options.slidesPerRow)+i));
if(h.get(g)){n.appendChild(h.get(g))
}}f.appendChild(n)
}d.appendChild(f)
}m.$slider.empty().append(d);
m.$slider.children().children().children().css({width:(100/m.options.slidesPerRow)+"%",display:"inline-block"})
}};
b.prototype.checkResponsive=function(h,j){var k=this,i,c,e,f=false;
var g=k.$slider.width();
var d=window.innerWidth||a(window).width();
if(k.respondTo==="window"){e=d
}else{if(k.respondTo==="slider"){e=g
}else{if(k.respondTo==="min"){e=Math.min(d,g)
}}}if(k.options.responsive&&k.options.responsive.length&&k.options.responsive!==null){c=null;
for(i in k.breakpoints){if(k.breakpoints.hasOwnProperty(i)){if(k.originalSettings.mobileFirst===false){if(e<k.breakpoints[i]){c=k.breakpoints[i]
}}else{if(e>k.breakpoints[i]){c=k.breakpoints[i]
}}}}if(c!==null){if(k.activeBreakpoint!==null){if(c!==k.activeBreakpoint||j){k.activeBreakpoint=c;
if(k.breakpointSettings[c]==="unslick"){k.unslick(c)
}else{k.options=a.extend({},k.originalSettings,k.breakpointSettings[c]);
if(h===true){k.currentSlide=k.options.initialSlide
}k.refresh(h)
}f=c
}}else{k.activeBreakpoint=c;
if(k.breakpointSettings[c]==="unslick"){k.unslick(c)
}else{k.options=a.extend({},k.originalSettings,k.breakpointSettings[c]);
if(h===true){k.currentSlide=k.options.initialSlide
}k.refresh(h)
}f=c
}}else{if(k.activeBreakpoint!==null){k.activeBreakpoint=null;
k.options=k.originalSettings;
if(h===true){k.currentSlide=k.options.initialSlide
}k.refresh(h);
f=c
}}if(!h&&f!==false){k.$slider.trigger("breakpoint",[k,f])
}}};
b.prototype.changeSlide=function(g,j){var e=this,c=a(g.currentTarget),i,f,h;
if(c.is("a")){g.preventDefault()
}if(!c.is("li")){c=c.closest("li")
}h=(e.slideCount%e.options.slidesToScroll!==0);
i=h?0:(e.slideCount-e.currentSlide)%e.options.slidesToScroll;
switch(g.data.message){case"previous":f=i===0?e.options.slidesToScroll:e.options.slidesToShow-i;
if(e.slideCount>e.options.slidesToShow){e.slideHandler(e.currentSlide-f,false,j)
}break;
case"next":f=i===0?e.options.slidesToScroll:i;
if(e.slideCount>e.options.slidesToShow){e.slideHandler(e.currentSlide+f,false,j)
}break;
case"index":var d=g.data.index===0?0:g.data.index||c.index()*e.options.slidesToScroll;
e.slideHandler(e.checkNavigable(d),false,j);
c.children().trigger("focus");
break;
default:return
}};
b.prototype.checkNavigable=function(d){var c=this,e,f;
e=c.getNavigableIndexes();
f=0;
if(d>e[e.length-1]){d=e[e.length-1]
}else{for(var g in e){if(d<e[g]){d=f;
break
}f=e[g]
}}return d
};
b.prototype.cleanUpEvents=function(){var c=this;
if(c.options.dots&&c.$dots!==null){a("li",c.$dots).off("click.slick",c.changeSlide).off("mouseenter.slick",a.proxy(c.interrupt,c,true)).off("mouseleave.slick",a.proxy(c.interrupt,c,false))
}c.$slider.off("focus.slick blur.slick");
if(c.options.arrows===true&&c.slideCount>c.options.slidesToShow){c.$prevArrow&&c.$prevArrow.off("click.slick",c.changeSlide);
c.$nextArrow&&c.$nextArrow.off("click.slick",c.changeSlide)
}c.$list.off("touchstart.slick mousedown.slick",c.swipeHandler);
c.$list.off("touchmove.slick mousemove.slick",c.swipeHandler);
c.$list.off("touchend.slick mouseup.slick",c.swipeHandler);
c.$list.off("touchcancel.slick mouseleave.slick",c.swipeHandler);
c.$list.off("click.slick",c.clickHandler);
a(document).off(c.visibilityChange,c.visibility);
c.cleanUpSlideEvents();
if(c.options.accessibility===true){c.$list.off("keydown.slick",c.keyHandler)
}if(c.options.focusOnSelect===true){a(c.$slideTrack).children().off("click.slick",c.selectHandler)
}a(window).off("orientationchange.slick.slick-"+c.instanceUid,c.orientationChange);
a(window).off("resize.slick.slick-"+c.instanceUid,c.resize);
a("[draggable!=true]",c.$slideTrack).off("dragstart",c.preventDefault);
a(window).off("load.slick.slick-"+c.instanceUid,c.setPosition);
a(document).off("ready.slick.slick-"+c.instanceUid,c.setPosition)
};
b.prototype.cleanUpSlideEvents=function(){var c=this;
c.$list.off("mouseenter.slick",a.proxy(c.interrupt,c,true));
c.$list.off("mouseleave.slick",a.proxy(c.interrupt,c,false))
};
b.prototype.cleanUpRows=function(){var d=this,c;
if(d.options.rows>1){c=d.$slides.children().children();
c.removeAttr("style");
d.$slider.empty().append(c)
}};
b.prototype.clickHandler=function(d){var c=this;
if(c.shouldClick===false){d.stopImmediatePropagation();
d.stopPropagation();
d.preventDefault()
}};
b.prototype.destroy=function(d){var c=this;
c.autoPlayClear();
c.touchObject={};
c.cleanUpEvents();
a(".slick-cloned",c.$slider).detach();
if(c.$dots){c.$dots.remove()
}if(c.$prevArrow&&c.$prevArrow.length){c.$prevArrow.removeClass("slick-disabled slick-arrow slick-hidden").removeAttr("aria-hidden aria-disabled tabindex").css("display","");
if(c.htmlExpr.test(c.options.prevArrow)){c.$prevArrow.remove()
}}if(c.$nextArrow&&c.$nextArrow.length){c.$nextArrow.removeClass("slick-disabled slick-arrow slick-hidden").removeAttr("aria-hidden aria-disabled tabindex").css("display","");
if(c.htmlExpr.test(c.options.nextArrow)){c.$nextArrow.remove()
}}if(c.$slides){c.$slides.removeClass("slick-slide slick-active slick-center slick-visible slick-current").removeAttr("aria-hidden").removeAttr("data-slick-index").each(function(){a(this).attr("style",a(this).data("originalStyling"))
});
c.$slideTrack.children(this.options.slide).detach();
c.$slideTrack.detach();
c.$list.detach();
c.$slider.append(c.$slides)
}c.cleanUpRows();
c.$slider.removeClass("slick-slider");
c.$slider.removeClass("slick-initialized");
c.$slider.removeClass("slick-dotted");
c.unslicked=true;
if(!d){c.$slider.trigger("destroy",[c])
}};
b.prototype.disableTransition=function(c){var d=this,e={};
e[d.transitionType]="";
if(d.options.fade===false){d.$slideTrack.css(e)
}else{d.$slides.eq(c).css(e)
}};
b.prototype.fadeSlide=function(d,e){var c=this;
if(c.cssTransitions===false){c.$slides.eq(d).css({zIndex:c.options.zIndex});
c.$slides.eq(d).animate({opacity:1},c.options.speed,c.options.easing,e)
}else{c.applyTransition(d);
c.$slides.eq(d).css({opacity:1,zIndex:c.options.zIndex});
if(e){setTimeout(function(){c.disableTransition(d);
e.call()
},c.options.speed)
}}};
b.prototype.fadeSlideOut=function(d){var c=this;
if(c.cssTransitions===false){c.$slides.eq(d).animate({opacity:0,zIndex:c.options.zIndex-2},c.options.speed,c.options.easing)
}else{c.applyTransition(d);
c.$slides.eq(d).css({opacity:0,zIndex:c.options.zIndex-2})
}};
b.prototype.filterSlides=b.prototype.slickFilter=function(d){var c=this;
if(d!==null){c.$slidesCache=c.$slides;
c.unload();
c.$slideTrack.children(this.options.slide).detach();
c.$slidesCache.filter(d).appendTo(c.$slideTrack);
c.reinit()
}};
b.prototype.focusHandler=function(){var c=this;
c.$slider.off("focus.slick blur.slick").on("focus.slick blur.slick","*:not(.slick-arrow)",function(e){e.stopImmediatePropagation();
var d=a(this);
setTimeout(function(){if(c.options.pauseOnFocus){c.focussed=d.is(":focus");
c.autoPlay()
}},0)
})
};
b.prototype.getCurrent=b.prototype.slickCurrentSlide=function(){var c=this;
return c.currentSlide
};
b.prototype.getDotCount=function(){var d=this;
var f=0;
var c=0;
var e=0;
if(d.options.infinite===true){while(f<d.slideCount){++e;
f=c+d.options.slidesToScroll;
c+=d.options.slidesToScroll<=d.options.slidesToShow?d.options.slidesToScroll:d.options.slidesToShow
}}else{if(d.options.centerMode===true){e=d.slideCount
}else{if(!d.options.asNavFor){e=1+Math.ceil((d.slideCount-d.options.slidesToShow)/d.options.slidesToScroll)
}else{while(f<d.slideCount){++e;
f=c+d.options.slidesToScroll;
c+=d.options.slidesToScroll<=d.options.slidesToShow?d.options.slidesToScroll:d.options.slidesToShow
}}}}return e-1
};
b.prototype.getLeft=function(f){var d=this,h,e,c=0,g;
d.slideOffset=0;
e=d.$slides.first().outerHeight(true);
if(d.options.infinite===true){if(d.slideCount>d.options.slidesToShow){d.slideOffset=(d.slideWidth*d.options.slidesToShow)*-1;
c=(e*d.options.slidesToShow)*-1
}if(d.slideCount%d.options.slidesToScroll!==0){if(f+d.options.slidesToScroll>d.slideCount&&d.slideCount>d.options.slidesToShow){if(f>d.slideCount){d.slideOffset=((d.options.slidesToShow-(f-d.slideCount))*d.slideWidth)*-1;
c=((d.options.slidesToShow-(f-d.slideCount))*e)*-1
}else{d.slideOffset=((d.slideCount%d.options.slidesToScroll)*d.slideWidth)*-1;
c=((d.slideCount%d.options.slidesToScroll)*e)*-1
}}}}else{if(f+d.options.slidesToShow>d.slideCount){d.slideOffset=((f+d.options.slidesToShow)-d.slideCount)*d.slideWidth;
c=((f+d.options.slidesToShow)-d.slideCount)*e
}}if(d.slideCount<=d.options.slidesToShow){d.slideOffset=0;
c=0
}if(d.options.centerMode===true&&d.options.infinite===true){d.slideOffset+=d.slideWidth*Math.floor(d.options.slidesToShow/2)-d.slideWidth
}else{if(d.options.centerMode===true){d.slideOffset=0;
d.slideOffset+=d.slideWidth*Math.floor(d.options.slidesToShow/2)
}}if(d.options.vertical===false){h=((f*d.slideWidth)*-1)+d.slideOffset
}else{h=((f*e)*-1)+c
}if(d.options.variableWidth===true){if(d.slideCount<=d.options.slidesToShow||d.options.infinite===false){g=d.$slideTrack.children(".slick-slide").eq(f)
}else{g=d.$slideTrack.children(".slick-slide").eq(f+d.options.slidesToShow)
}if(d.options.rtl===true){if(g[0]){h=(d.$slideTrack.width()-g[0].offsetLeft-g.width())*-1
}else{h=0
}}else{h=g[0]?g[0].offsetLeft*-1:0
}if(d.options.centerMode===true){if(d.slideCount<=d.options.slidesToShow||d.options.infinite===false){g=d.$slideTrack.children(".slick-slide").eq(f)
}else{g=d.$slideTrack.children(".slick-slide").eq(f+d.options.slidesToShow+1)
}if(d.options.rtl===true){if(g[0]){h=(d.$slideTrack.width()-g[0].offsetLeft-g.width())*-1
}else{h=0
}}else{h=g[0]?g[0].offsetLeft*-1:0
}h+=(d.$list.width()-g.outerWidth())/2
}}return h
};
b.prototype.getOption=b.prototype.slickGetOption=function(d){var c=this;
return c.options[d]
};
b.prototype.getNavigableIndexes=function(){var f=this,g=0,d=0,e=[],c;
if(f.options.infinite===false){c=f.slideCount
}else{g=f.options.slidesToScroll*-1;
d=f.options.slidesToScroll*-1;
c=f.slideCount*2
}while(g<c){e.push(g);
g=d+f.options.slidesToScroll;
d+=f.options.slidesToScroll<=f.options.slidesToShow?f.options.slidesToScroll:f.options.slidesToShow
}return e
};
b.prototype.getSlick=function(){return this
};
b.prototype.getSlideCount=function(){var e=this,d,f,c;
c=e.options.centerMode===true?e.slideWidth*Math.floor(e.options.slidesToShow/2):0;
if(e.options.swipeToSlide===true){e.$slideTrack.find(".slick-slide").each(function(h,g){if(g.offsetLeft-c+(a(g).outerWidth()/2)>(e.swipeLeft*-1)){f=g;
return false
}});
d=Math.abs(a(f).attr("data-slick-index")-e.currentSlide)||1;
return d
}else{return e.options.slidesToScroll
}};
b.prototype.goTo=b.prototype.slickGoTo=function(c,e){var d=this;
d.changeSlide({data:{message:"index",index:parseInt(c)}},e)
};
b.prototype.init=function(c){var d=this;
if(!a(d.$slider).hasClass("slick-initialized")){a(d.$slider).addClass("slick-initialized");
d.buildRows();
d.buildOut();
d.setProps();
d.startLoad();
d.loadSlider();
d.initializeEvents();
d.updateArrows();
d.updateDots();
d.checkResponsive(true);
d.focusHandler()
}if(c){d.$slider.trigger("init",[d])
}if(d.options.accessibility===true){d.initADA()
}if(d.options.autoplay){d.paused=false;
d.autoPlay()
}};
b.prototype.initADA=function(){var c=this;
c.$slides.add(c.$slideTrack.find(".slick-cloned")).attr({"aria-hidden":"true",tabindex:"-1"}).find("a, input, button, select").attr({tabindex:"-1"});
c.$slideTrack.attr("role","listbox");
c.$slides.not(c.$slideTrack.find(".slick-cloned")).each(function(d){a(this).attr({role:"option","aria-describedby":"slick-slide"+c.instanceUid+d+""})
});
if(c.$dots!==null){c.$dots.attr("role","tablist").find("li").each(function(d){a(this).attr({role:"presentation","aria-selected":"false","aria-controls":"navigation"+c.instanceUid+d+"",id:"slick-slide"+c.instanceUid+d+""})
}).first().attr("aria-selected","true").end().find("button").attr("role","button").end().closest("div").attr("role","toolbar")
}c.activateADA()
};
b.prototype.initArrowEvents=function(){var c=this;
if(c.options.arrows===true&&c.slideCount>c.options.slidesToShow){c.$prevArrow.off("click.slick").on("click.slick",{message:"previous"},c.changeSlide);
c.$nextArrow.off("click.slick").on("click.slick",{message:"next"},c.changeSlide)
}};
b.prototype.initDotEvents=function(){var c=this;
if(c.options.dots===true&&c.slideCount>c.options.slidesToShow){a("li",c.$dots).on("click.slick",{message:"index"},c.changeSlide)
}if(c.options.dots===true&&c.options.pauseOnDotsHover===true){a("li",c.$dots).on("mouseenter.slick",a.proxy(c.interrupt,c,true)).on("mouseleave.slick",a.proxy(c.interrupt,c,false))
}};
b.prototype.initSlideEvents=function(){var c=this;
if(c.options.pauseOnHover){c.$list.on("mouseenter.slick",a.proxy(c.interrupt,c,true));
c.$list.on("mouseleave.slick",a.proxy(c.interrupt,c,false))
}};
b.prototype.initializeEvents=function(){var c=this;
c.initArrowEvents();
c.initDotEvents();
c.initSlideEvents();
c.$list.on("touchstart.slick mousedown.slick",{action:"start"},c.swipeHandler);
c.$list.on("touchmove.slick mousemove.slick",{action:"move"},c.swipeHandler);
c.$list.on("touchend.slick mouseup.slick",{action:"end"},c.swipeHandler);
c.$list.on("touchcancel.slick mouseleave.slick",{action:"end"},c.swipeHandler);
c.$list.on("click.slick",c.clickHandler);
a(document).on(c.visibilityChange,a.proxy(c.visibility,c));
if(c.options.accessibility===true){c.$list.on("keydown.slick",c.keyHandler)
}if(c.options.focusOnSelect===true){a(c.$slideTrack).children().on("click.slick",c.selectHandler)
}a(window).on("orientationchange.slick.slick-"+c.instanceUid,a.proxy(c.orientationChange,c));
a(window).on("resize.slick.slick-"+c.instanceUid,a.proxy(c.resize,c));
a("[draggable!=true]",c.$slideTrack).on("dragstart",c.preventDefault);
a(window).on("load.slick.slick-"+c.instanceUid,c.setPosition);
a(document).on("ready.slick.slick-"+c.instanceUid,c.setPosition)
};
b.prototype.initUI=function(){var c=this;
if(c.options.arrows===true&&c.slideCount>c.options.slidesToShow){c.$prevArrow.show();
c.$nextArrow.show()
}if(c.options.dots===true&&c.slideCount>c.options.slidesToShow){c.$dots.show()
}};
b.prototype.keyHandler=function(d){var c=this;
if(!d.target.tagName.match("TEXTAREA|INPUT|SELECT")){if(d.keyCode===37&&c.options.accessibility===true){c.changeSlide({data:{message:c.options.rtl===true?"next":"previous"}})
}else{if(d.keyCode===39&&c.options.accessibility===true){c.changeSlide({data:{message:c.options.rtl===true?"previous":"next"}})
}}}};
b.prototype.lazyLoad=function(){var e=this,c,h,g,f;
function d(i){a("img[data-lazy]",i).each(function(){var k=a(this),l=a(this).attr("data-lazy"),j=document.createElement("img");
j.onload=function(){k.animate({opacity:0},100,function(){k.attr("src",l).animate({opacity:1},200,function(){k.removeAttr("data-lazy").removeClass("slick-loading")
});
e.$slider.trigger("lazyLoaded",[e,k,l])
})
};
j.onerror=function(){k.removeAttr("data-lazy").removeClass("slick-loading").addClass("slick-lazyload-error");
e.$slider.trigger("lazyLoadError",[e,k,l])
};
j.src=l
})
}if(e.options.centerMode===true){if(e.options.infinite===true){g=e.currentSlide+(e.options.slidesToShow/2+1);
f=g+e.options.slidesToShow+2
}else{g=Math.max(0,e.currentSlide-(e.options.slidesToShow/2+1));
f=2+(e.options.slidesToShow/2+1)+e.currentSlide
}}else{g=e.options.infinite?e.options.slidesToShow+e.currentSlide:e.currentSlide;
f=Math.ceil(g+e.options.slidesToShow);
if(e.options.fade===true){if(g>0){g--
}if(f<=e.slideCount){f++
}}}c=e.$slider.find(".slick-slide").slice(g,f);
d(c);
if(e.slideCount<=e.options.slidesToShow){h=e.$slider.find(".slick-slide");
d(h)
}else{if(e.currentSlide>=e.slideCount-e.options.slidesToShow){h=e.$slider.find(".slick-cloned").slice(0,e.options.slidesToShow);
d(h)
}else{if(e.currentSlide===0){h=e.$slider.find(".slick-cloned").slice(e.options.slidesToShow*-1);
d(h)
}}}};
b.prototype.loadSlider=function(){var c=this;
c.setPosition();
c.$slideTrack.css({opacity:1});
c.$slider.removeClass("slick-loading");
c.initUI();
if(c.options.lazyLoad==="progressive"){c.progressiveLazyLoad()
}};
b.prototype.next=b.prototype.slickNext=function(){var c=this;
c.changeSlide({data:{message:"next"}})
};
b.prototype.orientationChange=function(){var c=this;
c.checkResponsive();
c.setPosition()
};
b.prototype.pause=b.prototype.slickPause=function(){var c=this;
c.autoPlayClear();
c.paused=true
};
b.prototype.play=b.prototype.slickPlay=function(){var c=this;
c.autoPlay();
c.options.autoplay=true;
c.paused=false;
c.focussed=false;
c.interrupted=false
};
b.prototype.postSlide=function(d){var c=this;
if(!c.unslicked){c.$slider.trigger("afterChange",[c,d]);
c.animating=false;
c.setPosition();
c.swipeLeft=null;
if(c.options.autoplay){c.autoPlay()
}if(c.options.accessibility===true){c.initADA()
}}};
b.prototype.prev=b.prototype.slickPrev=function(){var c=this;
c.changeSlide({data:{message:"previous"}})
};
b.prototype.preventDefault=function(c){c.preventDefault()
};
b.prototype.progressiveLazyLoad=function(e){e=e||1;
var d=this,c=a("img[data-lazy]",d.$slider),g,h,f;
if(c.length){g=c.first();
h=g.attr("data-lazy");
f=document.createElement("img");
f.onload=function(){g.attr("src",h).removeAttr("data-lazy").removeClass("slick-loading");
if(d.options.adaptiveHeight===true){d.setPosition()
}d.$slider.trigger("lazyLoaded",[d,g,h]);
d.progressiveLazyLoad()
};
f.onerror=function(){if(e<3){setTimeout(function(){d.progressiveLazyLoad(e+1)
},500)
}else{g.removeAttr("data-lazy").removeClass("slick-loading").addClass("slick-lazyload-error");
d.$slider.trigger("lazyLoadError",[d,g,h]);
d.progressiveLazyLoad()
}};
f.src=h
}else{d.$slider.trigger("allImagesLoaded",[d])
}};
b.prototype.refresh=function(d){var e=this,f,c;
c=e.slideCount-e.options.slidesToShow;
if(!e.options.infinite&&(e.currentSlide>c)){e.currentSlide=c
}if(e.slideCount<=e.options.slidesToShow){e.currentSlide=0
}f=e.currentSlide;
e.destroy(true);
a.extend(e,e.initials,{currentSlide:f});
e.init();
if(!d){e.changeSlide({data:{message:"index",index:f}},false)
}};
b.prototype.registerBreakpoints=function(){var e=this,d,f,c,g=e.options.responsive||null;
if(a.type(g)==="array"&&g.length){e.respondTo=e.options.respondTo||"window";
for(d in g){c=e.breakpoints.length-1;
f=g[d].breakpoint;
if(g.hasOwnProperty(d)){while(c>=0){if(e.breakpoints[c]&&e.breakpoints[c]===f){e.breakpoints.splice(c,1)
}c--
}e.breakpoints.push(f);
e.breakpointSettings[f]=g[d].settings
}}e.breakpoints.sort(function(i,h){return(e.options.mobileFirst)?i-h:h-i
})
}};
b.prototype.reinit=function(){var c=this;
c.$slides=c.$slideTrack.children(c.options.slide).addClass("slick-slide");
c.slideCount=c.$slides.length;
if(c.currentSlide>=c.slideCount&&c.currentSlide!==0){c.currentSlide=c.currentSlide-c.options.slidesToScroll
}if(c.slideCount<=c.options.slidesToShow){c.currentSlide=0
}c.registerBreakpoints();
c.setProps();
c.setupInfinite();
c.buildArrows();
c.updateArrows();
c.initArrowEvents();
c.buildDots();
c.updateDots();
c.initDotEvents();
c.cleanUpSlideEvents();
c.initSlideEvents();
c.checkResponsive(false,true);
if(c.options.focusOnSelect===true){a(c.$slideTrack).children().on("click.slick",c.selectHandler)
}c.setSlideClasses(typeof c.currentSlide==="number"?c.currentSlide:0);
c.setPosition();
c.focusHandler();
c.paused=!c.options.autoplay;
c.autoPlay();
c.$slider.trigger("reInit",[c])
};
b.prototype.resize=function(){var c=this;
if(a(window).width()!==c.windowWidth){clearTimeout(c.windowDelay);
c.windowDelay=window.setTimeout(function(){c.windowWidth=a(window).width();
c.checkResponsive();
if(!c.unslicked){c.setPosition()
}},50)
}};
b.prototype.removeSlide=b.prototype.slickRemove=function(d,f,e){var c=this;
if(typeof(d)==="boolean"){f=d;
d=f===true?0:c.slideCount-1
}else{d=f===true?--d:d
}if(c.slideCount<1||d<0||d>c.slideCount-1){return false
}c.unload();
if(e===true){c.$slideTrack.children().remove()
}else{c.$slideTrack.children(this.options.slide).eq(d).remove()
}c.$slides=c.$slideTrack.children(this.options.slide);
c.$slideTrack.children(this.options.slide).detach();
c.$slideTrack.append(c.$slides);
c.$slidesCache=c.$slides;
c.reinit()
};
b.prototype.setCSS=function(d){var e=this,f={},c,g;
if(e.options.rtl===true){d=-d
}c=e.positionProp=="left"?Math.ceil(d)+"px":"0px";
g=e.positionProp=="top"?Math.ceil(d)+"px":"0px";
f[e.positionProp]=d;
if(e.transformsEnabled===false){e.$slideTrack.css(f)
}else{f={};
if(e.cssTransitions===false){f[e.animType]="translate("+c+", "+g+")";
e.$slideTrack.css(f)
}else{f[e.animType]="translate3d("+c+", "+g+", 0px)";
e.$slideTrack.css(f)
}}};
b.prototype.setDimensions=function(){var c=this;
if(c.options.vertical===false){if(c.options.centerMode===true){c.$list.css({padding:("0px "+c.options.centerPadding)})
}}else{c.$list.height(c.$slides.first().outerHeight(true)*c.options.slidesToShow);
if(c.options.centerMode===true){c.$list.css({padding:(c.options.centerPadding+" 0px")})
}}c.listWidth=c.$list.width();
c.listHeight=c.$list.height();
if(c.options.vertical===false&&c.options.variableWidth===false){c.slideWidth=Math.ceil(c.listWidth/c.options.slidesToShow);
c.$slideTrack.width(Math.ceil((c.slideWidth*c.$slideTrack.children(".slick-slide").length)))
}else{if(c.options.variableWidth===true){c.$slideTrack.width(5000*c.slideCount)
}else{c.slideWidth=Math.ceil(c.listWidth);
c.$slideTrack.height(Math.ceil((c.$slides.first().outerHeight(true)*c.$slideTrack.children(".slick-slide").length)))
}}var d=c.$slides.first().outerWidth(true)-c.$slides.first().width();
if(c.options.variableWidth===false){c.$slideTrack.children(".slick-slide").width(c.slideWidth-d)
}};
b.prototype.setFade=function(){var c=this,d;
c.$slides.each(function(e,f){d=(c.slideWidth*e)*-1;
if(c.options.rtl===true){a(f).css({position:"relative",right:d,top:0,zIndex:c.options.zIndex-2,opacity:0})
}else{a(f).css({position:"relative",left:d,top:0,zIndex:c.options.zIndex-2,opacity:0})
}});
c.$slides.eq(c.currentSlide).css({zIndex:c.options.zIndex-1,opacity:1})
};
b.prototype.setHeight=function(){var d=this;
if(d.options.slidesToShow===1&&d.options.adaptiveHeight===true&&d.options.vertical===false){var c=d.$slides.eq(d.currentSlide).outerHeight(true);
d.$list.css("height",c)
}};
b.prototype.setOption=b.prototype.slickSetOption=function(){var d=this,c,h,g,i,f=false,e;
if(a.type(arguments[0])==="object"){g=arguments[0];
f=arguments[1];
e="multiple"
}else{if(a.type(arguments[0])==="string"){g=arguments[0];
i=arguments[1];
f=arguments[2];
if(arguments[0]==="responsive"&&a.type(arguments[1])==="array"){e="responsive"
}else{if(typeof arguments[1]!=="undefined"){e="single"
}}}}if(e==="single"){d.options[g]=i
}else{if(e==="multiple"){a.each(g,function(j,k){d.options[j]=k
})
}else{if(e==="responsive"){for(h in i){if(a.type(d.options.responsive)!=="array"){d.options.responsive=[i[h]]
}else{c=d.options.responsive.length-1;
while(c>=0){if(d.options.responsive[c].breakpoint===i[h].breakpoint){d.options.responsive.splice(c,1)
}c--
}d.options.responsive.push(i[h])
}}}}}if(f){d.unload();
d.reinit()
}};
b.prototype.setPosition=function(){var c=this;
c.setDimensions();
c.setHeight();
if(c.options.fade===false){c.setCSS(c.getLeft(c.currentSlide))
}else{c.setFade()
}c.$slider.trigger("setPosition",[c])
};
b.prototype.setProps=function(){var d=this,c=document.body.style;
d.positionProp=d.options.vertical===true?"top":"left";
if(d.positionProp==="top"){d.$slider.addClass("slick-vertical")
}else{d.$slider.removeClass("slick-vertical")
}if(c.WebkitTransition!==undefined||c.MozTransition!==undefined||c.msTransition!==undefined){if(d.options.useCSS===true){d.cssTransitions=true
}}if(d.options.fade){if(typeof d.options.zIndex==="number"){if(d.options.zIndex<3){d.options.zIndex=3
}}else{d.options.zIndex=d.defaults.zIndex
}}if(c.OTransform!==undefined){d.animType="OTransform";
d.transformType="-o-transform";
d.transitionType="OTransition";
if(c.perspectiveProperty===undefined&&c.webkitPerspective===undefined){d.animType=false
}}if(c.MozTransform!==undefined){d.animType="MozTransform";
d.transformType="-moz-transform";
d.transitionType="MozTransition";
if(c.perspectiveProperty===undefined&&c.MozPerspective===undefined){d.animType=false
}}if(c.webkitTransform!==undefined){d.animType="webkitTransform";
d.transformType="-webkit-transform";
d.transitionType="webkitTransition";
if(c.perspectiveProperty===undefined&&c.webkitPerspective===undefined){d.animType=false
}}if(c.msTransform!==undefined){d.animType="msTransform";
d.transformType="-ms-transform";
d.transitionType="msTransition";
if(c.msTransform===undefined){d.animType=false
}}if(c.transform!==undefined&&d.animType!==false){d.animType="transform";
d.transformType="transform";
d.transitionType="transition"
}d.transformsEnabled=d.options.useTransform&&(d.animType!==null&&d.animType!==false)
};
b.prototype.setSlideClasses=function(f){var e=this,c,d,h,g;
d=e.$slider.find(".slick-slide").removeClass("slick-active slick-center slick-current").attr("aria-hidden","true");
e.$slides.eq(f).addClass("slick-current");
if(e.options.centerMode===true){c=Math.floor(e.options.slidesToShow/2);
if(e.options.infinite===true){if(f>=c&&f<=(e.slideCount-1)-c){e.$slides.slice(f-c,f+c+1).addClass("slick-active").attr("aria-hidden","false")
}else{h=e.options.slidesToShow+f;
d.slice(h-c+1,h+c+2).addClass("slick-active").attr("aria-hidden","false")
}if(f===0){d.eq(d.length-1-e.options.slidesToShow).addClass("slick-center")
}else{if(f===e.slideCount-1){d.eq(e.options.slidesToShow).addClass("slick-center")
}}}e.$slides.eq(f).addClass("slick-center")
}else{if(f>=0&&f<=(e.slideCount-e.options.slidesToShow)){e.$slides.slice(f,f+e.options.slidesToShow).addClass("slick-active").attr("aria-hidden","false")
}else{if(d.length<=e.options.slidesToShow){d.addClass("slick-active").attr("aria-hidden","false")
}else{g=e.slideCount%e.options.slidesToShow;
h=e.options.infinite===true?e.options.slidesToShow+f:f;
if(e.options.slidesToShow==e.options.slidesToScroll&&(e.slideCount-f)<e.options.slidesToShow){d.slice(h-(e.options.slidesToShow-g),h+g).addClass("slick-active").attr("aria-hidden","false")
}else{d.slice(h,h+e.options.slidesToShow).addClass("slick-active").attr("aria-hidden","false")
}}}}if(e.options.lazyLoad==="ondemand"){e.lazyLoad()
}};
b.prototype.setupInfinite=function(){var c=this,d,f,e;
if(c.options.fade===true){c.options.centerMode=false
}if(c.options.infinite===true&&c.options.fade===false){f=null;
if(c.slideCount>c.options.slidesToShow){if(c.options.centerMode===true){e=c.options.slidesToShow+1
}else{e=c.options.slidesToShow
}for(d=c.slideCount;
d>(c.slideCount-e);
d-=1){f=d-1;
a(c.$slides[f]).clone(true).attr("id","").attr("data-slick-index",f-c.slideCount).prependTo(c.$slideTrack).addClass("slick-cloned")
}for(d=0;
d<e;
d+=1){f=d;
a(c.$slides[f]).clone(true).attr("id","").attr("data-slick-index",f+c.slideCount).appendTo(c.$slideTrack).addClass("slick-cloned")
}c.$slideTrack.find(".slick-cloned").find("[id]").each(function(){a(this).attr("id","")
})
}}};
b.prototype.interrupt=function(c){var d=this;
if(!c){d.autoPlay()
}d.interrupted=c
};
b.prototype.selectHandler=function(f){var e=this;
var d=a(f.target).is(".slick-slide")?a(f.target):a(f.target).parents(".slick-slide");
var c=parseInt(d.attr("data-slick-index"));
if(!c){c=0
}if(e.slideCount<=e.options.slidesToShow){e.setSlideClasses(c);
e.asNavFor(c);
return
}e.slideHandler(c)
};
b.prototype.slideHandler=function(f,i,e){var d,l,h,j,g=null,k=this,c;
i=i||false;
if(k.animating===true&&k.options.waitForAnimate===true){return
}if(k.options.fade===true&&k.currentSlide===f){return
}if(k.slideCount<=k.options.slidesToShow){return
}if(i===false){k.asNavFor(f)
}d=f;
g=k.getLeft(d);
j=k.getLeft(k.currentSlide);
k.currentLeft=k.swipeLeft===null?j:k.swipeLeft;
if(k.options.infinite===false&&k.options.centerMode===false&&(f<0||f>k.getDotCount()*k.options.slidesToScroll)){if(k.options.fade===false){d=k.currentSlide;
if(e!==true){k.animateSlide(j,function(){k.postSlide(d)
})
}else{k.postSlide(d)
}}return
}else{if(k.options.infinite===false&&k.options.centerMode===true&&(f<0||f>(k.slideCount-k.options.slidesToScroll))){if(k.options.fade===false){d=k.currentSlide;
if(e!==true){k.animateSlide(j,function(){k.postSlide(d)
})
}else{k.postSlide(d)
}}return
}}if(k.options.autoplay){clearInterval(k.autoPlayTimer)
}if(d<0){if(k.slideCount%k.options.slidesToScroll!==0){l=k.slideCount-(k.slideCount%k.options.slidesToScroll)
}else{l=k.slideCount+d
}}else{if(d>=k.slideCount){if(k.slideCount%k.options.slidesToScroll!==0){l=0
}else{l=d-k.slideCount
}}else{l=d
}}k.animating=true;
k.$slider.trigger("beforeChange",[k,k.currentSlide,l]);
h=k.currentSlide;
k.currentSlide=l;
k.setSlideClasses(k.currentSlide);
if(k.options.asNavFor){c=k.getNavTarget();
c=c.slick("getSlick");
if(c.slideCount<=c.options.slidesToShow){c.setSlideClasses(k.currentSlide)
}}k.updateDots();
k.updateArrows();
if(k.options.fade===true){if(e!==true){k.fadeSlideOut(h);
k.fadeSlide(l,function(){k.postSlide(l)
})
}else{k.postSlide(l)
}k.animateHeight();
return
}if(e!==true){k.animateSlide(g,function(){k.postSlide(l)
})
}else{k.postSlide(l)
}};
b.prototype.startLoad=function(){var c=this;
if(c.options.arrows===true&&c.slideCount>c.options.slidesToShow){c.$prevArrow.hide();
c.$nextArrow.hide()
}if(c.options.dots===true&&c.slideCount>c.options.slidesToShow){c.$dots.hide()
}c.$slider.addClass("slick-loading")
};
b.prototype.swipeDirection=function(){var c,f,e,g,d=this;
c=d.touchObject.startX-d.touchObject.curX;
f=d.touchObject.startY-d.touchObject.curY;
e=Math.atan2(f,c);
g=Math.round(e*180/Math.PI);
if(g<0){g=360-Math.abs(g)
}if((g<=45)&&(g>=0)){return(d.options.rtl===false?"left":"right")
}if((g<=360)&&(g>=315)){return(d.options.rtl===false?"left":"right")
}if((g>=135)&&(g<=225)){return(d.options.rtl===false?"right":"left")
}if(d.options.verticalSwiping===true){if((g>=35)&&(g<=135)){return"down"
}else{return"up"
}}return"vertical"
};
b.prototype.swipeEnd=function(e){var d=this,c,f;
d.dragging=false;
d.interrupted=false;
d.shouldClick=(d.touchObject.swipeLength>10)?false:true;
if(d.touchObject.curX===undefined){return false
}if(d.touchObject.edgeHit===true){d.$slider.trigger("edge",[d,d.swipeDirection()])
}if(d.touchObject.swipeLength>=d.touchObject.minSwipe){f=d.swipeDirection();
switch(f){case"left":case"down":c=d.options.swipeToSlide?d.checkNavigable(d.currentSlide+d.getSlideCount()):d.currentSlide+d.getSlideCount();
d.currentDirection=0;
break;
case"right":case"up":c=d.options.swipeToSlide?d.checkNavigable(d.currentSlide-d.getSlideCount()):d.currentSlide-d.getSlideCount();
d.currentDirection=1;
break;
default:}if(f!="vertical"){d.slideHandler(c);
d.touchObject={};
d.$slider.trigger("swipe",[d,f])
}}else{if(d.touchObject.startX!==d.touchObject.curX){d.slideHandler(d.currentSlide);
d.touchObject={}
}}};
b.prototype.swipeHandler=function(d){var c=this;
if((c.options.swipe===false)||("ontouchend" in document&&c.options.swipe===false)){return
}else{if(c.options.draggable===false&&d.type.indexOf("mouse")!==-1){return
}}c.touchObject.fingerCount=d.originalEvent&&d.originalEvent.touches!==undefined?d.originalEvent.touches.length:1;
c.touchObject.minSwipe=c.listWidth/c.options.touchThreshold;
if(c.options.verticalSwiping===true){c.touchObject.minSwipe=c.listHeight/c.options.touchThreshold
}switch(d.data.action){case"start":c.swipeStart(d);
break;
case"move":c.swipeMove(d);
break;
case"end":c.swipeEnd(d);
break
}};
b.prototype.swipeMove=function(f){var e=this,j=false,h,d,g,c,i;
i=f.originalEvent!==undefined?f.originalEvent.touches:null;
if(!e.dragging||i&&i.length!==1){return false
}h=e.getLeft(e.currentSlide);
e.touchObject.curX=i!==undefined?i[0].pageX:f.clientX;
e.touchObject.curY=i!==undefined?i[0].pageY:f.clientY;
e.touchObject.swipeLength=Math.round(Math.sqrt(Math.pow(e.touchObject.curX-e.touchObject.startX,2)));
if(e.options.verticalSwiping===true){e.touchObject.swipeLength=Math.round(Math.sqrt(Math.pow(e.touchObject.curY-e.touchObject.startY,2)))
}d=e.swipeDirection();
if(d==="vertical"){return
}if(f.originalEvent!==undefined&&e.touchObject.swipeLength>4){f.preventDefault()
}c=(e.options.rtl===false?1:-1)*(e.touchObject.curX>e.touchObject.startX?1:-1);
if(e.options.verticalSwiping===true){c=e.touchObject.curY>e.touchObject.startY?1:-1
}g=e.touchObject.swipeLength;
e.touchObject.edgeHit=false;
if(e.options.infinite===false){if((e.currentSlide===0&&d==="right")||(e.currentSlide>=e.getDotCount()&&d==="left")){g=e.touchObject.swipeLength*e.options.edgeFriction;
e.touchObject.edgeHit=true
}}if(e.options.vertical===false){e.swipeLeft=h+g*c
}else{e.swipeLeft=h+(g*(e.$list.height()/e.listWidth))*c
}if(e.options.verticalSwiping===true){e.swipeLeft=h+g*c
}if(e.options.fade===true||e.options.touchMove===false){return false
}if(e.animating===true){e.swipeLeft=null;
return false
}e.setCSS(e.swipeLeft)
};
b.prototype.swipeStart=function(d){var c=this,e;
c.interrupted=true;
if(c.touchObject.fingerCount!==1||c.slideCount<=c.options.slidesToShow){c.touchObject={};
return false
}if(d.originalEvent!==undefined&&d.originalEvent.touches!==undefined){e=d.originalEvent.touches[0]
}c.touchObject.startX=c.touchObject.curX=e!==undefined?e.pageX:d.clientX;
c.touchObject.startY=c.touchObject.curY=e!==undefined?e.pageY:d.clientY;
c.dragging=true
};
b.prototype.unfilterSlides=b.prototype.slickUnfilter=function(){var c=this;
if(c.$slidesCache!==null){c.unload();
c.$slideTrack.children(this.options.slide).detach();
c.$slidesCache.appendTo(c.$slideTrack);
c.reinit()
}};
b.prototype.unload=function(){var c=this;
a(".slick-cloned",c.$slider).remove();
if(c.$dots){c.$dots.remove()
}if(c.$prevArrow&&c.htmlExpr.test(c.options.prevArrow)){c.$prevArrow.remove()
}if(c.$nextArrow&&c.htmlExpr.test(c.options.nextArrow)){c.$nextArrow.remove()
}c.$slides.removeClass("slick-slide slick-active slick-visible slick-current").attr("aria-hidden","true").css("width","")
};
b.prototype.unslick=function(d){var c=this;
c.$slider.trigger("unslick",[c,d]);
c.destroy()
};
b.prototype.updateArrows=function(){var d=this,c;
c=Math.floor(d.options.slidesToShow/2);
if(d.options.arrows===true&&d.slideCount>d.options.slidesToShow&&!d.options.infinite){d.$prevArrow.removeClass("slick-disabled").attr("aria-disabled","false");
d.$nextArrow.removeClass("slick-disabled").attr("aria-disabled","false");
if(d.currentSlide===0){d.$prevArrow.addClass("slick-disabled").attr("aria-disabled","true");
d.$nextArrow.removeClass("slick-disabled").attr("aria-disabled","false")
}else{if(d.currentSlide>=d.slideCount-d.options.slidesToShow&&d.options.centerMode===false){d.$nextArrow.addClass("slick-disabled").attr("aria-disabled","true");
d.$prevArrow.removeClass("slick-disabled").attr("aria-disabled","false")
}else{if(d.currentSlide>=d.slideCount-1&&d.options.centerMode===true){d.$nextArrow.addClass("slick-disabled").attr("aria-disabled","true");
d.$prevArrow.removeClass("slick-disabled").attr("aria-disabled","false")
}}}}};
b.prototype.updateDots=function(){var c=this;
if(c.$dots!==null){c.$dots.find("li").removeClass("slick-active").attr("aria-hidden","true");
c.$dots.find("li").eq(Math.floor(c.currentSlide/c.options.slidesToScroll)).addClass("slick-active").attr("aria-hidden","false")
}};
b.prototype.visibility=function(){var c=this;
if(c.options.autoplay){if(document[c.hidden]){c.interrupted=true
}else{c.interrupted=false
}}};
a.fn.slick=function(){var f=this,h=arguments[0],e=Array.prototype.slice.call(arguments,1),c=f.length,g,d;
for(g=0;
g<c;
g++){if(typeof h=="object"||typeof h=="undefined"){f[g].slick=new b(f[g],h)
}else{d=f[g].slick[h].apply(f[g].slick,e)
}if(typeof d!="undefined"){return d
}}return f
};
a.fn.sameHeight=function(){var c=this;
var e=[];
c.each(function(){var f=a(this).height();
e.push(f)
});
var d=Math.max.apply(null,e);
c.each(function(){a(this).height(d)
})
}
}));
(function(a,b,c){(function(){var n="2.2.0",j="datepicker",h=".datepicker-here",e,g,l=false,m='<div class="datepicker"><i class="datepicker--pointer"></i><nav class="datepicker--nav"></nav><div class="datepicker--content"></div></div>',d={classes:"",inline:false,language:"ru",resetPosition:true,startDate:new Date(),firstDay:"",weekends:[6,0],dateFormat:"",altField:"",altFieldDateFormat:"@",toggleSelected:true,keyboardNav:true,position:"bottom left",offset:12,view:"days",minView:"days",showOtherMonths:true,selectOtherMonths:true,moveToOtherMonthsOnSelect:true,showOtherYears:true,selectOtherYears:true,moveToOtherYearsOnSelect:true,minDate:"",maxDate:"",disableNavWhenOutOfRange:true,multipleDates:false,multipleDatesSeparator:",",range:false,todayButton:false,clearButton:false,showEvent:"focus",autoClose:false,monthsField:"monthsShort",prevHtml:'<svg><path d="M 17,12 l -5,5 l 5,5"></path></svg>',nextHtml:'<svg><path d="M 14,12 l 5,5 l -5,5"></path></svg>',navTitles:{days:"MM, <i>yyyy</i>",months:"yyyy",years:"yyyy1 - yyyy2"},timepicker:false,onlyTimepicker:false,dateTimeSeparator:" ",timeFormat:"",minHours:0,maxHours:24,minMinutes:0,maxMinutes:59,hoursStep:1,minutesStep:1,onSelect:"",onShow:"",onHide:"",onChangeMonth:"",onChangeYear:"",onChangeDecade:"",onChangeView:"",onRenderCell:""},k={ctrlRight:[17,39],ctrlUp:[17,38],ctrlLeft:[17,37],ctrlDown:[17,40],shiftRight:[16,39],shiftUp:[16,38],shiftLeft:[16,37],shiftDown:[16,40],altUp:[18,38],altRight:[18,39],altLeft:[18,37],altDown:[18,40],ctrlShiftUp:[16,17,38]},i;
var f=function(p,o){this.el=p;
this.$el=b(p);
this.opts=b.extend(true,{},d,o,this.$el.data());
if(e==c){e=b("body")
}if(!this.opts.startDate){this.opts.startDate=new Date()
}if(this.el.nodeName=="INPUT"){this.elIsInput=true
}if(this.opts.altField){this.$altField=typeof this.opts.altField=="string"?b(this.opts.altField):this.opts.altField
}this.inited=false;
this.visible=false;
this.silent=false;
this.currentDate=this.opts.startDate;
this.currentView=this.opts.view;
this._createShortCuts();
this.selectedDates=[];
this.views={};
this.keys=[];
this.minRange="";
this.maxRange="";
this._prevOnSelectValue="";
this.init()
};
i=f;
i.prototype={VERSION:n,viewIndexes:["days","months","years"],init:function(){if(!l&&!this.opts.inline&&this.elIsInput){this._buildDatepickersContainer()
}this._buildBaseHtml();
this._defineLocale(this.opts.language);
this._syncWithMinMaxDates();
if(this.elIsInput){if(!this.opts.inline){this._setPositionClasses(this.opts.position);
this._bindEvents()
}if(this.opts.keyboardNav&&!this.opts.onlyTimepicker){this._bindKeyboardEvents()
}this.$datepicker.on("mousedown",this._onMouseDownDatepicker.bind(this));
this.$datepicker.on("mouseup",this._onMouseUpDatepicker.bind(this))
}if(this.opts.classes){this.$datepicker.addClass(this.opts.classes)
}if(this.opts.timepicker){this.timepicker=new b.fn.datepicker.Timepicker(this,this.opts);
this._bindTimepickerEvents()
}if(this.opts.onlyTimepicker){this.$datepicker.addClass("-only-timepicker-")
}this.views[this.currentView]=new b.fn.datepicker.Body(this,this.currentView,this.opts);
this.views[this.currentView].show();
this.nav=new b.fn.datepicker.Navigation(this,this.opts);
this.view=this.currentView;
this.$el.on("clickCell.adp",this._onClickCell.bind(this));
this.$datepicker.on("mouseenter",".datepicker--cell",this._onMouseEnterCell.bind(this));
this.$datepicker.on("mouseleave",".datepicker--cell",this._onMouseLeaveCell.bind(this));
this.inited=true
},_createShortCuts:function(){this.minDate=this.opts.minDate?this.opts.minDate:new Date(-8639999913600000);
this.maxDate=this.opts.maxDate?this.opts.maxDate:new Date(8639999913600000)
},_bindEvents:function(){this.$el.on(this.opts.showEvent+".adp",this._onShowEvent.bind(this));
this.$el.on("mouseup.adp",this._onMouseUpEl.bind(this));
this.$el.on("blur.adp",this._onBlur.bind(this));
this.$el.on("keyup.adp",this._onKeyUpGeneral.bind(this));
b(a).on("resize.adp",this._onResize.bind(this));
b("body").on("mouseup.adp",this._onMouseUpBody.bind(this))
},_bindKeyboardEvents:function(){this.$el.on("keydown.adp",this._onKeyDown.bind(this));
this.$el.on("keyup.adp",this._onKeyUp.bind(this));
this.$el.on("hotKey.adp",this._onHotKey.bind(this))
},_bindTimepickerEvents:function(){this.$el.on("timeChange.adp",this._onTimeChange.bind(this))
},isWeekend:function(o){return this.opts.weekends.indexOf(o)!==-1
},_defineLocale:function(o){if(typeof o=="string"){this.loc=b.fn.datepicker.language[o];
if(!this.loc){console.warn("Can't find language \""+o+'" in Datepicker.language, will use "ru" instead');
this.loc=b.extend(true,{},b.fn.datepicker.language.ru)
}this.loc=b.extend(true,{},b.fn.datepicker.language.ru,b.fn.datepicker.language[o])
}else{this.loc=b.extend(true,{},b.fn.datepicker.language.ru,o)
}if(this.opts.dateFormat){this.loc.dateFormat=this.opts.dateFormat
}if(this.opts.timeFormat){this.loc.timeFormat=this.opts.timeFormat
}if(this.opts.firstDay!==""){this.loc.firstDay=this.opts.firstDay
}if(this.opts.timepicker){this.loc.dateFormat=[this.loc.dateFormat,this.loc.timeFormat].join(this.opts.dateTimeSeparator)
}if(this.opts.onlyTimepicker){this.loc.dateFormat=this.loc.timeFormat
}var p=this._getWordBoundaryRegExp;
if(this.loc.timeFormat.match(p("aa"))||this.loc.timeFormat.match(p("AA"))){this.ampm=true
}},_buildDatepickersContainer:function(){l=true;
e.append('<div class="datepickers-container" id="datepickers-container"></div>');
g=b("#datepickers-container")
},_buildBaseHtml:function(){var p,o=b('<div class="datepicker-inline">');
if(this.el.nodeName=="INPUT"){if(!this.opts.inline){p=g
}else{p=o.insertAfter(this.$el)
}}else{p=o.appendTo(this.$el)
}this.$datepicker=b(m).appendTo(p);
this.$content=b(".datepicker--content",this.$datepicker);
this.$nav=b(".datepicker--nav",this.$datepicker)
},_triggerOnChange:function(){if(!this.selectedDates.length){if(this._prevOnSelectValue===""){return
}this._prevOnSelectValue="";
return this.opts.onSelect("","",this)
}var o=this.selectedDates,s=i.getParsedDate(o[0]),p,r=this,q=new Date(s.year,s.month,s.date,s.hours,s.minutes);
p=o.map(function(t){return r.formatDate(r.loc.dateFormat,t)
}).join(this.opts.multipleDatesSeparator);
if(this.opts.multipleDates||this.opts.range){q=o.map(function(t){var u=i.getParsedDate(t);
return new Date(u.year,u.month,u.date,u.hours,u.minutes)
})
}this._prevOnSelectValue=p;
this.opts.onSelect(p,q,this)
},next:function(){var q=this.parsedDate,p=this.opts;
switch(this.view){case"days":this.date=new Date(q.year,q.month+1,1);
if(p.onChangeMonth){p.onChangeMonth(this.parsedDate.month,this.parsedDate.year)
}break;
case"months":this.date=new Date(q.year+1,q.month,1);
if(p.onChangeYear){p.onChangeYear(this.parsedDate.year)
}break;
case"years":this.date=new Date(q.year+10,0,1);
if(p.onChangeDecade){p.onChangeDecade(this.curDecade)
}break
}},prev:function(){var q=this.parsedDate,p=this.opts;
switch(this.view){case"days":this.date=new Date(q.year,q.month-1,1);
if(p.onChangeMonth){p.onChangeMonth(this.parsedDate.month,this.parsedDate.year)
}break;
case"months":this.date=new Date(q.year-1,q.month,1);
if(p.onChangeYear){p.onChangeYear(this.parsedDate.year)
}break;
case"years":this.date=new Date(q.year-10,0,1);
if(p.onChangeDecade){p.onChangeDecade(this.curDecade)
}break
}},formatDate:function(t,q){q=q||this.date;
var A=t,p=this._getWordBoundaryRegExp,x=this.loc,y=i.getLeadingZeroNum,z=i.getDecade(q),u=i.getParsedDate(q),s=u.fullHours,w=u.hours,v=t.match(p("aa"))||t.match(p("AA")),r="am",o;
if(this.opts.timepicker&&this.timepicker&&v){o=this.timepicker._getValidHoursFromDate(q,v);
s=y(o.hours);
w=o.hours;
r=o.dayPeriod
}switch(true){case /@/.test(A):A=A.replace(/@/,q.getTime());
case /aa/.test(A):A=A.replace(p("aa"),r);
case /AA/.test(A):A=A.replace(p("AA"),r.toUpperCase());
case /dd/.test(A):A=A.replace(p("dd"),u.fullDate);
case /d/.test(A):A=A.replace(p("d"),u.date);
case /DD/.test(A):A=A.replace(p("DD"),x.days[u.day]);
case /D/.test(A):A=A.replace(p("D"),x.daysShort[u.day]);
case /mm/.test(A):A=A.replace(p("mm"),u.fullMonth);
case /m/.test(A):A=A.replace(p("m"),u.month+1);
case /MM/.test(A):A=A.replace(p("MM"),this.loc.months[u.month]);
case /M/.test(A):A=A.replace(p("M"),x.monthsShort[u.month]);
case /ii/.test(A):A=A.replace(p("ii"),u.fullMinutes);
case /i/.test(A):A=A.replace(p("i"),u.minutes);
case /hh/.test(A):A=A.replace(p("hh"),s);
case /h/.test(A):A=A.replace(p("h"),w);
case /yyyy/.test(A):A=A.replace(p("yyyy"),u.year);
case /yyyy1/.test(A):A=A.replace(p("yyyy1"),z[0]);
case /yyyy2/.test(A):A=A.replace(p("yyyy2"),z[1]);
case /yy/.test(A):A=A.replace(p("yy"),u.year.toString().slice(-2))
}return A
},_getWordBoundaryRegExp:function(o){return new RegExp("\\b(?=[a-zA-Z0-9Ã¡Ã¤Ã¶Ã¼ÃºÃÃŸÃ‰Ã„Ã–ÃœÃš<])"+o+"(?![>a-zA-Z0-9Ã¡Ã¤Ã¶Ã¼ÃÃŸÃ‰Ã„Ã–ÃœÃš])")
},selectDate:function(q){var u=this,s=u.opts,t=u.parsedDate,r=u.selectedDates,o=r.length,p="";
if(Array.isArray(q)){q.forEach(function(v){u.selectDate(v)
});
return
}if(!(q instanceof Date)){return
}this.lastSelectedDate=q;
if(this.timepicker){this.timepicker._setTime(q)
}u._trigger("selectDate",q);
if(this.timepicker){q.setHours(this.timepicker.hours);
q.setMinutes(this.timepicker.minutes)
}if(u.view=="days"){if(q.getMonth()!=t.month&&s.moveToOtherMonthsOnSelect){p=new Date(q.getFullYear(),q.getMonth(),1)
}}if(u.view=="years"){if(q.getFullYear()!=t.year&&s.moveToOtherYearsOnSelect){p=new Date(q.getFullYear(),0,1)
}}if(p){u.silent=true;
u.date=p;
u.silent=false;
u.nav._render()
}if(s.multipleDates&&!s.range){if(o===s.multipleDates){return
}if(!u._isSelected(q)){u.selectedDates.push(q)
}}else{if(s.range){if(o==2){u.selectedDates=[q];
u.minRange=q;
u.maxRange=""
}else{if(o==1){u.selectedDates.push(q);
if(!u.maxRange){u.maxRange=q
}else{u.minRange=q
}if(i.bigger(u.maxRange,u.minRange)){u.maxRange=u.minRange;
u.minRange=q
}u.selectedDates=[u.minRange,u.maxRange]
}else{u.selectedDates=[q];
u.minRange=q
}}}else{u.selectedDates=[q]
}}u._setInputValue();
if(s.onSelect){u._triggerOnChange()
}if(s.autoClose&&!this.timepickerIsActive){if(!s.multipleDates&&!s.range){u.hide()
}else{if(s.range&&u.selectedDates.length==2){u.hide()
}}}u.views[this.currentView]._render()
},removeDate:function(o){var p=this.selectedDates,q=this;
if(!(o instanceof Date)){return
}return p.some(function(r,s){if(i.isSame(r,o)){p.splice(s,1);
if(!q.selectedDates.length){q.minRange="";
q.maxRange="";
q.lastSelectedDate=""
}else{q.lastSelectedDate=q.selectedDates[q.selectedDates.length-1]
}q.views[q.currentView]._render();
q._setInputValue();
if(q.opts.onSelect){q._triggerOnChange()
}return true
}})
},today:function(){this.silent=true;
this.view=this.opts.minView;
this.silent=false;
this.date=new Date();
if(this.opts.todayButton instanceof Date){this.selectDate(this.opts.todayButton)
}},clear:function(){this.selectedDates=[];
this.minRange="";
this.maxRange="";
this.views[this.currentView]._render();
this._setInputValue();
if(this.opts.onSelect){this._triggerOnChange()
}},update:function(r,q){var o=arguments.length,p=this.lastSelectedDate;
if(o==2){this.opts[r]=q
}else{if(o==1&&typeof r=="object"){this.opts=b.extend(true,this.opts,r)
}}this._createShortCuts();
this._syncWithMinMaxDates();
this._defineLocale(this.opts.language);
this.nav._addButtonsIfNeed();
if(!this.opts.onlyTimepicker){this.nav._render()
}this.views[this.currentView]._render();
if(this.elIsInput&&!this.opts.inline){this._setPositionClasses(this.opts.position);
if(this.visible){this.setPosition(this.opts.position)
}}if(this.opts.classes){this.$datepicker.addClass(this.opts.classes)
}if(this.opts.onlyTimepicker){this.$datepicker.addClass("-only-timepicker-")
}if(this.opts.timepicker){if(p){this.timepicker._handleDate(p)
}this.timepicker._updateRanges();
this.timepicker._updateCurrentTime();
if(p){p.setHours(this.timepicker.hours);
p.setMinutes(this.timepicker.minutes)
}}this._setInputValue();
return this
},_syncWithMinMaxDates:function(){var o=this.date.getTime();
this.silent=true;
if(this.minTime>o){this.date=this.minDate
}if(this.maxTime<o){this.date=this.maxDate
}this.silent=false
},_isSelected:function(q,o){var p=false;
this.selectedDates.some(function(r){if(i.isSame(r,q,o)){p=r;
return true
}});
return p
},_setInputValue:function(){var t=this,q=t.opts,s=t.loc.dateFormat,p=q.altFieldDateFormat,r=t.selectedDates.map(function(u){return t.formatDate(s,u)
}),o;
if(q.altField&&t.$altField.length){o=this.selectedDates.map(function(u){return t.formatDate(p,u)
});
o=o.join(this.opts.multipleDatesSeparator);
this.$altField.val(o)
}r=r.join(this.opts.multipleDatesSeparator);
this.$el.val(r)
},_isInRange:function(q,u){var p=q.getTime(),t=i.getParsedDate(q),r=i.getParsedDate(this.minDate),v=i.getParsedDate(this.maxDate),o=new Date(t.year,t.month,r.date).getTime(),w=new Date(t.year,t.month,v.date).getTime(),s={day:p>=this.minTime&&p<=this.maxTime,month:o>=this.minTime&&w<=this.maxTime,year:t.year>=r.year&&t.year<=v.year};
return u?s[u]:s.day
},_getDimensions:function(o){var p=o.offset();
return{width:o.outerWidth(),height:o.outerHeight(),left:p.left,top:p.top}
},_getDateFromCell:function(o){var p=this.parsedDate,r=o.data("year")||p.year,s=o.data("month")==c?p.month:o.data("month"),q=o.data("date")||1;
return new Date(r,s,q)
},_setPositionClasses:function(r){r=r.split(" ");
var o=r[0],q=r[1],p="datepicker -"+o+"-"+q+"- -from-"+o+"-";
if(this.visible){p+=" active"
}this.$datepicker.removeAttr("class").addClass(p)
},setPosition:function(r){r=r||this.opts.position;
var v=this._getDimensions(this.$el),s=this._getDimensions(this.$datepicker),t=r.split(" "),u,o,q=this.opts.offset,p=t[0],w=t[1];
switch(p){case"top":u=v.top-s.height-q;
break;
case"right":o=v.left+v.width+q;
break;
case"bottom":u=v.top+v.height+q;
break;
case"left":o=v.left-s.width-q;
break
}switch(w){case"top":u=v.top;
break;
case"right":o=v.left+v.width-s.width;
break;
case"bottom":u=v.top+v.height-s.height;
break;
case"left":o=v.left;
break;
case"center":if(/left|right/.test(p)){u=v.top+v.height/2-s.height/2
}else{o=v.left+v.width/2-s.width/2
}}this.$datepicker.css({left:o,top:u})
},show:function(){var o=this.opts.onShow;
this.setPosition(this.opts.position);
this.$datepicker.addClass("active").show();
this.visible=true;
if(o){this._bindVisionEvents(o)
}},hide:function(){var o=this.opts.onHide;
this.$datepicker.removeClass("active").hide();
this.focused="";
this.keys=[];
this.inFocus=false;
this.visible=false;
this.$el.blur();
if(o){this._bindVisionEvents(o)
}},down:function(o){this._changeView(o,"down")
},up:function(o){this._changeView(o,"up")
},_bindVisionEvents:function(o){this.$datepicker.off("transitionend.dp");
o(this,false);
this.$datepicker.one("transitionend.dp",o.bind(this,this,true))
},_changeView:function(p,o){p=p||this.focused||this.date;
var q=o=="up"?this.viewIndex+1:this.viewIndex-1;
if(q>2){q=2
}if(q<0){q=0
}this.silent=true;
this.date=new Date(p.getFullYear(),p.getMonth(),1);
this.silent=false;
this.view=this.viewIndexes[q]
},_handleHotKey:function(B){var u=i.getParsedDate(this._getFocusedDate()),r,p=this.opts,A,t,q=false,v=false,z=false,x=u.year,s=u.month,w=u.date;
switch(B){case"ctrlRight":case"ctrlUp":s+=1;
q=true;
break;
case"ctrlLeft":case"ctrlDown":s-=1;
q=true;
break;
case"shiftRight":case"shiftUp":v=true;
x+=1;
break;
case"shiftLeft":case"shiftDown":v=true;
x-=1;
break;
case"altRight":case"altUp":z=true;
x+=10;
break;
case"altLeft":case"altDown":z=true;
x-=10;
break;
case"ctrlShiftUp":this.up();
break
}t=i.getDaysCount(new Date(x,s));
A=new Date(x,s,w);
if(t<w){w=t
}if(A.getTime()<this.minTime){A=this.minDate
}else{if(A.getTime()>this.maxTime){A=this.maxDate
}}this.focused=A;
r=i.getParsedDate(A);
if(q&&p.onChangeMonth){p.onChangeMonth(r.month,r.year)
}if(v&&p.onChangeYear){p.onChangeYear(r.year)
}if(z&&p.onChangeDecade){p.onChangeDecade(this.curDecade)
}},_registerKey:function(o){var p=this.keys.some(function(q){return q==o
});
if(!p){this.keys.push(o)
}},_unRegisterKey:function(p){var o=this.keys.indexOf(p);
this.keys.splice(o,1)
},_isHotKeyPressed:function(){var o,r=false,s=this,p=this.keys.sort();
for(var q in k){o=k[q];
if(p.length!=o.length){continue
}if(o.every(function(u,t){return u==p[t]
})){s._trigger("hotKey",q);
r=true
}}return r
},_trigger:function(p,o){this.$el.trigger(p,o)
},_focusNextCell:function(r,q){q=q||this.cellType;
var p=i.getParsedDate(this._getFocusedDate()),u=p.year,o=p.month,t=p.date;
if(this._isHotKeyPressed()){return
}switch(r){case 37:q=="day"?(t-=1):"";
q=="month"?(o-=1):"";
q=="year"?(u-=1):"";
break;
case 38:q=="day"?(t-=7):"";
q=="month"?(o-=3):"";
q=="year"?(u-=4):"";
break;
case 39:q=="day"?(t+=1):"";
q=="month"?(o+=1):"";
q=="year"?(u+=1):"";
break;
case 40:q=="day"?(t+=7):"";
q=="month"?(o+=3):"";
q=="year"?(u+=4):"";
break
}var s=new Date(u,o,t);
if(s.getTime()<this.minTime){s=this.minDate
}else{if(s.getTime()>this.maxTime){s=this.maxDate
}}this.focused=s
},_getFocusedDate:function(){var o=this.focused||this.selectedDates[this.selectedDates.length-1],p=this.parsedDate;
if(!o){switch(this.view){case"days":o=new Date(p.year,p.month,new Date().getDate());
break;
case"months":o=new Date(p.year,p.month,1);
break;
case"years":o=new Date(p.year,0,1);
break
}}return o
},_getCell:function(p,q){q=q||this.cellType;
var s=i.getParsedDate(p),o='.datepicker--cell[data-year="'+s.year+'"]',r;
switch(q){case"month":o='[data-month="'+s.month+'"]';
break;
case"day":o+='[data-month="'+s.month+'"][data-date="'+s.date+'"]';
break
}r=this.views[this.currentView].$el.find(o);
return r.length?r:b("")
},destroy:function(){var o=this;
o.$el.off(".adp").data("datepicker","");
o.selectedDates=[];
o.focused="";
o.views={};
o.keys=[];
o.minRange="";
o.maxRange="";
if(o.opts.inline||!o.elIsInput){o.$datepicker.closest(".datepicker-inline").remove()
}else{o.$datepicker.remove()
}},_handleAlreadySelectedDates:function(o,p){if(this.opts.range){if(!this.opts.toggleSelected){if(this.selectedDates.length!=2){this._trigger("clickCell",p)
}}else{this.removeDate(p)
}}else{if(this.opts.toggleSelected){this.removeDate(p)
}}if(!this.opts.toggleSelected){this.lastSelectedDate=o;
if(this.opts.timepicker){this.timepicker._setTime(o);
this.timepicker.update()
}}},_onShowEvent:function(o){if(!this.visible){this.show()
}},_onBlur:function(){if(!this.inFocus&&this.visible){this.hide()
}},_onMouseDownDatepicker:function(o){this.inFocus=true
},_onMouseUpDatepicker:function(o){this.inFocus=false;
o.originalEvent.inFocus=true;
if(!o.originalEvent.timepickerFocus){this.$el.focus()
}},_onKeyUpGeneral:function(o){var p=this.$el.val();
if(!p){this.clear()
}},_onResize:function(){if(this.visible){this.setPosition()
}},_onMouseUpBody:function(o){if(o.originalEvent.inFocus){return
}if(this.visible&&!this.inFocus){this.hide()
}},_onMouseUpEl:function(o){o.originalEvent.inFocus=true;
setTimeout(this._onKeyUpGeneral.bind(this),4)
},_onKeyDown:function(q){var p=q.which;
this._registerKey(p);
if(p>=37&&p<=40){q.preventDefault();
this._focusNextCell(p)
}if(p==13){if(this.focused){if(this._getCell(this.focused).hasClass("-disabled-")){return
}if(this.view!=this.opts.minView){this.down()
}else{var o=this._isSelected(this.focused,this.cellType);
if(!o){if(this.timepicker){this.focused.setHours(this.timepicker.hours);
this.focused.setMinutes(this.timepicker.minutes)
}this.selectDate(this.focused);
return
}this._handleAlreadySelectedDates(o,this.focused)
}}}if(p==27){this.hide()
}},_onKeyUp:function(p){var o=p.which;
this._unRegisterKey(o)
},_onHotKey:function(p,o){this._handleHotKey(o)
},_onMouseEnterCell:function(q){var p=b(q.target).closest(".datepicker--cell"),o=this._getDateFromCell(p);
this.silent=true;
if(this.focused){this.focused=""
}p.addClass("-focus-");
this.focused=o;
this.silent=false;
if(this.opts.range&&this.selectedDates.length==1){this.minRange=this.selectedDates[0];
this.maxRange="";
if(i.less(this.minRange,this.focused)){this.maxRange=this.minRange;
this.minRange=""
}this.views[this.currentView]._update()
}},_onMouseLeaveCell:function(p){var o=b(p.target).closest(".datepicker--cell");
o.removeClass("-focus-");
this.silent=true;
this.focused="";
this.silent=false
},_onTimeChange:function(t,s,o){var p=new Date(),r=this.selectedDates,q=false;
if(r.length){q=true;
p=this.lastSelectedDate
}p.setHours(s);
p.setMinutes(o);
if(!q&&!this._getCell(p).hasClass("-disabled-")){this.selectDate(p)
}else{this._setInputValue();
if(this.opts.onSelect){this._triggerOnChange()
}}},_onClickCell:function(p,o){if(this.timepicker){o.setHours(this.timepicker.hours);
o.setMinutes(this.timepicker.minutes)
}this.selectDate(o)
},set focused(p){if(!p&&this.focused){var o=this._getCell(this.focused);
if(o.length){o.removeClass("-focus-")
}}this._focused=p;
if(this.opts.range&&this.selectedDates.length==1){this.minRange=this.selectedDates[0];
this.maxRange="";
if(i.less(this.minRange,this._focused)){this.maxRange=this.minRange;
this.minRange=""
}}if(this.silent){return
}this.date=p
},get focused(){return this._focused
},get parsedDate(){return i.getParsedDate(this.date)
},set date(o){if(!(o instanceof Date)){return
}this.currentDate=o;
if(this.inited&&!this.silent){this.views[this.view]._render();
this.nav._render();
if(this.visible&&this.elIsInput){this.setPosition()
}}return o
},get date(){return this.currentDate
},set view(o){this.viewIndex=this.viewIndexes.indexOf(o);
if(this.viewIndex<0){return
}this.prevView=this.currentView;
this.currentView=o;
if(this.inited){if(!this.views[o]){this.views[o]=new b.fn.datepicker.Body(this,o,this.opts)
}else{this.views[o]._render()
}this.views[this.prevView].hide();
this.views[o].show();
this.nav._render();
if(this.opts.onChangeView){this.opts.onChangeView(o)
}if(this.elIsInput&&this.visible){this.setPosition()
}}return o
},get view(){return this.currentView
},get cellType(){return this.view.substring(0,this.view.length-1)
},get minTime(){var o=i.getParsedDate(this.minDate);
return new Date(o.year,o.month,o.date).getTime()
},get maxTime(){var o=i.getParsedDate(this.maxDate);
return new Date(o.year,o.month,o.date).getTime()
},get curDecade(){return i.getDecade(this.date)
}};
i.getDaysCount=function(o){return new Date(o.getFullYear(),o.getMonth()+1,0).getDate()
};
i.getParsedDate=function(o){return{year:o.getFullYear(),month:o.getMonth(),fullMonth:(o.getMonth()+1)<10?"0"+(o.getMonth()+1):o.getMonth()+1,date:o.getDate(),fullDate:o.getDate()<10?"0"+o.getDate():o.getDate(),day:o.getDay(),hours:o.getHours(),fullHours:o.getHours()<10?"0"+o.getHours():o.getHours(),minutes:o.getMinutes(),fullMinutes:o.getMinutes()<10?"0"+o.getMinutes():o.getMinutes()}
};
i.getDecade=function(o){var p=Math.floor(o.getFullYear()/10)*10;
return[p,p+9]
};
i.template=function(p,o){return p.replace(/#\{([\w]+)\}/g,function(r,q){if(o[q]||o[q]===0){return o[q]
}})
};
i.isSame=function(u,s,p){if(!u||!s){return false
}var r=i.getParsedDate(u),q=i.getParsedDate(s),o=p?p:"day",t={day:r.date==q.date&&r.month==q.month&&r.year==q.year,month:r.month==q.month&&r.year==q.year,year:r.year==q.year};
return t[o]
};
i.less=function(p,o,q){if(!p||!o){return false
}return o.getTime()<p.getTime()
};
i.bigger=function(p,o,q){if(!p||!o){return false
}return o.getTime()>p.getTime()
};
i.getLeadingZeroNum=function(o){return parseInt(o)<10?"0"+o:o
};
i.resetTime=function(o){if(typeof o!="object"){return
}o=i.getParsedDate(o);
return new Date(o.year,o.month,o.date)
};
b.fn.datepicker=function(o){return this.each(function(){if(!b.data(this,j)){b.data(this,j,new f(this,o))
}else{var p=b.data(this,j);
p.opts=b.extend(true,p.opts,o);
p.update()
}})
};
b.fn.datepicker.Constructor=f;
b.fn.datepicker.language={ru:{days:["Ð’Ð¾ÑÐºÑ€ÐµÑÐµÐ½ÑŒÐµ","ÐŸÐ¾Ð½ÐµÐ´ÐµÐ»ÑŒÐ½Ð¸Ðº","Ð’Ñ‚Ð¾Ñ€Ð½Ð¸Ðº","Ð¡Ñ€ÐµÐ´Ð°","Ð§ÐµÑ‚Ð²ÐµÑ€Ð³","ÐŸÑÑ‚Ð½Ð¸Ñ†Ð°","Ð¡ÑƒÐ±Ð±Ð¾Ñ‚Ð°"],daysShort:["Ð’Ð¾Ñ","ÐŸÐ¾Ð½","Ð’Ñ‚Ð¾","Ð¡Ñ€Ðµ","Ð§ÐµÑ‚","ÐŸÑÑ‚","Ð¡ÑƒÐ±"],daysMin:["Ð’Ñ","ÐŸÐ½","Ð’Ñ‚","Ð¡Ñ€","Ð§Ñ‚","ÐŸÑ‚","Ð¡Ð±"],months:["Ð¯Ð½Ð²Ð°Ñ€ÑŒ","Ð¤ÐµÐ²Ñ€Ð°Ð»ÑŒ","ÐœÐ°Ñ€Ñ‚","ÐÐ¿Ñ€ÐµÐ»ÑŒ","ÐœÐ°Ð¹","Ð˜ÑŽÐ½ÑŒ","Ð˜ÑŽÐ»ÑŒ","ÐÐ²Ð³ÑƒÑÑ‚","Ð¡ÐµÐ½Ñ‚ÑÐ±Ñ€ÑŒ","ÐžÐºÑ‚ÑÐ±Ñ€ÑŒ","ÐÐ¾ÑÐ±Ñ€ÑŒ","Ð”ÐµÐºÐ°Ð±Ñ€ÑŒ"],monthsShort:["Ð¯Ð½Ð²","Ð¤ÐµÐ²","ÐœÐ°Ñ€","ÐÐ¿Ñ€","ÐœÐ°Ð¹","Ð˜ÑŽÐ½","Ð˜ÑŽÐ»","ÐÐ²Ð³","Ð¡ÐµÐ½","ÐžÐºÑ‚","ÐÐ¾Ñ","Ð”ÐµÐº"],today:"Ð¡ÐµÐ³Ð¾Ð´Ð½Ñ",clear:"ÐžÑ‡Ð¸ÑÑ‚Ð¸Ñ‚ÑŒ",dateFormat:"dd.mm.yyyy",timeFormat:"hh:ii",firstDay:1}};
b(function(){b(h).datepicker()
})
})();
(function(){var d={days:'<div class="datepicker--days datepicker--body"><div class="datepicker--days-names"></div><div class="datepicker--cells datepicker--cells-days"></div></div>',months:'<div class="datepicker--months datepicker--body"><div class="datepicker--cells datepicker--cells-months"></div></div>',years:'<div class="datepicker--years datepicker--body"><div class="datepicker--cells datepicker--cells-years"></div></div>'},e=b.fn.datepicker,f=e.Constructor;
e.Body=function(i,g,h){this.d=i;
this.type=g;
this.opts=h;
this.$el=b("");
if(this.opts.onlyTimepicker){return
}this.init()
};
e.Body.prototype={init:function(){this._buildBaseHtml();
this._render();
this._bindEvents()
},_bindEvents:function(){this.$el.on("click",".datepicker--cell",b.proxy(this._onClickCell,this))
},_buildBaseHtml:function(){this.$el=b(d[this.type]).appendTo(this.d.$content);
this.$names=b(".datepicker--days-names",this.$el);
this.$cells=b(".datepicker--cells",this.$el)
},_getDayNamesHtml:function(g,k,j,h){k=k!=c?k:g;
j=j?j:"";
h=h!=c?h:0;
if(h>7){return j
}if(k==7){return this._getDayNamesHtml(g,0,j,++h)
}j+='<div class="datepicker--day-name'+(this.d.isWeekend(k)?" -weekend-":"")+'">'+this.d.loc.daysMin[k]+"</div>";
return this._getDayNamesHtml(g,++k,j,++h)
},_getCellContents:function(k,p){var l="datepicker--cell datepicker--cell-"+p,i=new Date(),q=this.d,m=f.resetTime(q.minRange),j=f.resetTime(q.maxRange),g=q.opts,o=f.getParsedDate(k),h={},n=o.date;
switch(p){case"day":if(q.isWeekend(o.day)){l+=" -weekend-"
}if(o.month!=this.d.parsedDate.month){l+=" -other-month-";
if(!g.selectOtherMonths){l+=" -disabled-"
}if(!g.showOtherMonths){n=""
}}break;
case"month":n=q.loc[q.opts.monthsField][o.month];
break;
case"year":var r=q.curDecade;
n=o.year;
if(o.year<r[0]||o.year>r[1]){l+=" -other-decade-";
if(!g.selectOtherYears){l+=" -disabled-"
}if(!g.showOtherYears){n=""
}}break
}if(g.onRenderCell){h=g.onRenderCell(k,p)||{};
n=h.html?h.html:n;
l+=h.classes?" "+h.classes:""
}if(g.range){if(f.isSame(m,k,p)){l+=" -range-from-"
}if(f.isSame(j,k,p)){l+=" -range-to-"
}if(q.selectedDates.length==1&&q.focused){if((f.bigger(m,k)&&f.less(q.focused,k))||(f.less(j,k)&&f.bigger(q.focused,k))){l+=" -in-range-"
}if(f.less(j,k)&&f.isSame(q.focused,k)){l+=" -range-from-"
}if(f.bigger(m,k)&&f.isSame(q.focused,k)){l+=" -range-to-"
}}else{if(q.selectedDates.length==2){if(f.bigger(m,k)&&f.less(j,k)){l+=" -in-range-"
}}}}if(f.isSame(i,k,p)){l+=" -current-"
}if(q.focused&&f.isSame(k,q.focused,p)){l+=" -focus-"
}if(q._isSelected(k,p)){l+=" -selected-"
}if(!q._isInRange(k,p)||h.disabled){l+=" -disabled-"
}return{html:n,classes:l}
},_getDaysHtml:function(j){var h=f.getDaysCount(j),l=new Date(j.getFullYear(),j.getMonth(),1).getDay(),p=new Date(j.getFullYear(),j.getMonth(),h).getDay(),q=l-this.d.loc.firstDay,g=6-p+this.d.loc.firstDay;
q=q<0?q+7:q;
g=g>6?g-7:g;
var t=-q+1,k,s,o="";
for(var n=t,r=h+g;
n<=r;
n++){s=j.getFullYear();
k=j.getMonth();
o+=this._getDayHtml(new Date(s,k,n))
}return o
},_getDayHtml:function(g){var h=this._getCellContents(g,"day");
return'<div class="'+h.classes+'" data-date="'+g.getDate()+'" data-month="'+g.getMonth()+'" data-year="'+g.getFullYear()+'">'+h.html+"</div>"
},_getMonthsHtml:function(g){var j="",k=f.getParsedDate(g),h=0;
while(h<12){j+=this._getMonthHtml(new Date(k.year,h));
h++
}return j
},_getMonthHtml:function(g){var h=this._getCellContents(g,"month");
return'<div class="'+h.classes+'" data-month="'+g.getMonth()+'">'+h.html+"</div>"
},_getYearsHtml:function(g){var m=f.getParsedDate(g),l=f.getDecade(g),k=l[0]-1,j="",h=k;
for(h;
h<=l[1]+1;
h++){j+=this._getYearHtml(new Date(h,0))
}return j
},_getYearHtml:function(g){var h=this._getCellContents(g,"year");
return'<div class="'+h.classes+'" data-year="'+g.getFullYear()+'">'+h.html+"</div>"
},_renderTypes:{days:function(){var h=this._getDayNamesHtml(this.d.loc.firstDay),g=this._getDaysHtml(this.d.currentDate);
this.$cells.html(g);
this.$names.html(h)
},months:function(){var g=this._getMonthsHtml(this.d.currentDate);
this.$cells.html(g)
},years:function(){var g=this._getYearsHtml(this.d.currentDate);
this.$cells.html(g)
}},_render:function(){if(this.opts.onlyTimepicker){return
}this._renderTypes[this.type].bind(this)()
},_update:function(){var g=b(".datepicker--cell",this.$cells),k=this,i,j,h;
g.each(function(l,m){j=b(this);
h=k.d._getDateFromCell(b(this));
i=k._getCellContents(h,k.d.cellType);
j.attr("class",i.classes)
})
},show:function(){if(this.opts.onlyTimepicker){return
}this.$el.addClass("active");
this.acitve=true
},hide:function(){this.$el.removeClass("active");
this.active=false
},_handleClick:function(k){var h=k.data("date")||1,l=k.data("month")||0,j=k.data("year")||this.d.parsedDate.year,m=this.d;
if(m.view!=this.opts.minView){m.down(new Date(j,l,h));
return
}var i=new Date(j,l,h),g=this.d._isSelected(i,this.d.cellType);
if(!g){m._trigger("clickCell",i);
return
}m._handleAlreadySelectedDates.bind(m,g,i)()
},_onClickCell:function(h){var g=b(h.target).closest(".datepicker--cell");
if(g.hasClass("-disabled-")){return
}this._handleClick.bind(this)(g)
}}
})();
(function(){var f='<div class="datepicker--nav-action" data-action="prev">#{prevHtml}</div><div class="datepicker--nav-title">#{title}</div><div class="datepicker--nav-action" data-action="next">#{nextHtml}</div>',e='<div class="datepicker--buttons"></div>',d='<span class="datepicker--button" data-action="#{action}">#{label}</span>',g=b.fn.datepicker,h=g.Constructor;
g.Navigation=function(j,i){this.d=j;
this.opts=i;
this.$buttonsContainer="";
this.init()
};
g.Navigation.prototype={init:function(){this._buildBaseHtml();
this._bindEvents()
},_bindEvents:function(){this.d.$nav.on("click",".datepicker--nav-action",b.proxy(this._onClickNavButton,this));
this.d.$nav.on("click",".datepicker--nav-title",b.proxy(this._onClickNavTitle,this));
this.d.$datepicker.on("click",".datepicker--button",b.proxy(this._onClickNavButton,this))
},_buildBaseHtml:function(){if(!this.opts.onlyTimepicker){this._render()
}this._addButtonsIfNeed()
},_addButtonsIfNeed:function(){if(this.opts.todayButton){this._addButton("today")
}if(this.opts.clearButton){this._addButton("clear")
}},_render:function(){var j=this._getTitle(this.d.currentDate),i=h.template(f,b.extend({title:j},this.opts));
this.d.$nav.html(i);
if(this.d.view=="years"){b(".datepicker--nav-title",this.d.$nav).addClass("-disabled-")
}this.setNavStatus()
},_getTitle:function(i){return this.d.formatDate(this.opts.navTitles[this.d.view],i)
},_addButton:function(j){if(!this.$buttonsContainer.length){this._addButtonsContainer()
}var k={action:j,label:this.d.loc[j]},i=h.template(d,k);
if(b("[data-action="+j+"]",this.$buttonsContainer).length){return
}this.$buttonsContainer.append(i)
},_addButtonsContainer:function(){this.d.$datepicker.append(e);
this.$buttonsContainer=b(".datepicker--buttons",this.d.$datepicker)
},setNavStatus:function(){if(!(this.opts.minDate||this.opts.maxDate)||!this.opts.disableNavWhenOutOfRange){return
}var j=this.d.parsedDate,i=j.month,l=j.year,k=j.date;
switch(this.d.view){case"days":if(!this.d._isInRange(new Date(l,i-1,k),"month")){this._disableNav("prev")
}if(!this.d._isInRange(new Date(l,i+1,k),"month")){this._disableNav("next")
}break;
case"months":if(!this.d._isInRange(new Date(l-1,i,k),"year")){this._disableNav("prev")
}if(!this.d._isInRange(new Date(l+1,i,k),"year")){this._disableNav("next")
}break;
case"years":if(!this.d._isInRange(new Date(l-10,i,k),"year")){this._disableNav("prev")
}if(!this.d._isInRange(new Date(l+10,i,k),"year")){this._disableNav("next")
}break
}},_disableNav:function(i){b('[data-action="'+i+'"]',this.d.$nav).addClass("-disabled-")
},_activateNav:function(i){b('[data-action="'+i+'"]',this.d.$nav).removeClass("-disabled-")
},_onClickNavButton:function(k){var i=b(k.target).closest("[data-action]"),j=i.data("action");
this.d[j]()
},_onClickNavTitle:function(i){if(b(i.target).hasClass("-disabled-")){return
}if(this.d.view=="days"){return this.d.view="months"
}this.d.view="years"
}}
})();
(function(){var d='<div class="datepicker--time"><div class="datepicker--time-current">   <span class="datepicker--time-current-hours">#{hourVisible}</span>   <span class="datepicker--time-current-colon">:</span>   <span class="datepicker--time-current-minutes">#{minValue}</span></div><div class="datepicker--time-sliders">   <div class="datepicker--time-row">      <input type="range" name="hours" value="#{hourValue}" min="#{hourMin}" max="#{hourMax}" step="#{hourStep}"/>   </div>   <div class="datepicker--time-row">      <input type="range" name="minutes" value="#{minValue}" min="#{minMin}" max="#{minMax}" step="#{minStep}"/>   </div></div></div>',e=b.fn.datepicker,f=e.Constructor;
e.Timepicker=function(h,g){this.d=h;
this.opts=g;
this.init()
};
e.Timepicker.prototype={init:function(){var g="input";
this._setTime(this.d.date);
this._buildHTML();
if(navigator.userAgent.match(/trident/gi)){g="change"
}this.d.$el.on("selectDate",this._onSelectDate.bind(this));
this.$ranges.on(g,this._onChangeRange.bind(this));
this.$ranges.on("mouseup",this._onMouseUpRange.bind(this));
this.$ranges.on("mousemove focus ",this._onMouseEnterRange.bind(this));
this.$ranges.on("mouseout blur",this._onMouseOutRange.bind(this))
},_setTime:function(h){var g=f.getParsedDate(h);
this._handleDate(h);
this.hours=g.hours<this.minHours?this.minHours:g.hours;
this.minutes=g.minutes<this.minMinutes?this.minMinutes:g.minutes
},_setMinTimeFromDate:function(g){this.minHours=g.getHours();
this.minMinutes=g.getMinutes();
if(this.d.lastSelectedDate){if(this.d.lastSelectedDate.getHours()>g.getHours()){this.minMinutes=this.opts.minMinutes
}}},_setMaxTimeFromDate:function(g){this.maxHours=g.getHours();
this.maxMinutes=g.getMinutes();
if(this.d.lastSelectedDate){if(this.d.lastSelectedDate.getHours()<g.getHours()){this.maxMinutes=this.opts.maxMinutes
}}},_setDefaultMinMaxTime:function(){var h=23,g=59,i=this.opts;
this.minHours=i.minHours<0||i.minHours>h?0:i.minHours;
this.minMinutes=i.minMinutes<0||i.minMinutes>g?0:i.minMinutes;
this.maxHours=i.maxHours<0||i.maxHours>h?h:i.maxHours;
this.maxMinutes=i.maxMinutes<0||i.maxMinutes>g?g:i.maxMinutes
},_validateHoursMinutes:function(g){if(this.hours<this.minHours){this.hours=this.minHours
}else{if(this.hours>this.maxHours){this.hours=this.maxHours
}}if(this.minutes<this.minMinutes){this.minutes=this.minMinutes
}else{if(this.minutes>this.maxMinutes){this.minutes=this.maxMinutes
}}},_buildHTML:function(){var g=f.getLeadingZeroNum,i={hourMin:this.minHours,hourMax:g(this.maxHours),hourStep:this.opts.hoursStep,hourValue:this.hours,hourVisible:g(this.displayHours),minMin:this.minMinutes,minMax:g(this.maxMinutes),minStep:this.opts.minutesStep,minValue:g(this.minutes)},h=f.template(d,i);
this.$timepicker=b(h).appendTo(this.d.$datepicker);
this.$ranges=b('[type="range"]',this.$timepicker);
this.$hours=b('[name="hours"]',this.$timepicker);
this.$minutes=b('[name="minutes"]',this.$timepicker);
this.$hoursText=b(".datepicker--time-current-hours",this.$timepicker);
this.$minutesText=b(".datepicker--time-current-minutes",this.$timepicker);
if(this.d.ampm){this.$ampm=b('<span class="datepicker--time-current-ampm">').appendTo(b(".datepicker--time-current",this.$timepicker)).html(this.dayPeriod);
this.$timepicker.addClass("-am-pm-")
}},_updateCurrentTime:function(){var i=f.getLeadingZeroNum(this.displayHours),g=f.getLeadingZeroNum(this.minutes);
this.$hoursText.html(i);
this.$minutesText.html(g);
if(this.d.ampm){this.$ampm.html(this.dayPeriod)
}},_updateRanges:function(){this.$hours.attr({min:this.minHours,max:this.maxHours}).val(this.hours);
this.$minutes.attr({min:this.minMinutes,max:this.maxMinutes}).val(this.minutes)
},_handleDate:function(g){this._setDefaultMinMaxTime();
if(g){if(f.isSame(g,this.d.opts.minDate)){this._setMinTimeFromDate(this.d.opts.minDate)
}else{if(f.isSame(g,this.d.opts.maxDate)){this._setMaxTimeFromDate(this.d.opts.maxDate)
}}}this._validateHoursMinutes(g)
},update:function(){this._updateRanges();
this._updateCurrentTime()
},_getValidHoursFromDate:function(k,j){var l=k,h=k;
if(k instanceof Date){l=f.getParsedDate(k);
h=l.hours
}var g=j||this.d.ampm,i="am";
if(g){switch(true){case h==0:h=12;
break;
case h==12:i="pm";
break;
case h>11:h=h-12;
i="pm";
break;
default:break
}}return{hours:h,dayPeriod:i}
},set hours(h){this._hours=h;
var g=this._getValidHoursFromDate(h);
this.displayHours=g.hours;
this.dayPeriod=g.dayPeriod
},get hours(){return this._hours
},_onChangeRange:function(i){var g=b(i.target),h=g.attr("name");
this.d.timepickerIsActive=true;
this[h]=g.val();
this._updateCurrentTime();
this.d._trigger("timeChange",[this.hours,this.minutes]);
this._handleDate(this.d.lastSelectedDate);
this.update()
},_onSelectDate:function(h,g){this._handleDate(g);
this.update()
},_onMouseEnterRange:function(h){var g=b(h.target).attr("name");
b(".datepicker--time-current-"+g,this.$timepicker).addClass("-focus-")
},_onMouseOutRange:function(h){var g=b(h.target).attr("name");
if(this.d.inFocus){return
}b(".datepicker--time-current-"+g,this.$timepicker).removeClass("-focus-")
},_onMouseUpRange:function(g){this.d.timepickerIsActive=false
}}
})()
})(window,jQuery);
(function(a){a.fn.datepicker.language.en={days:["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],daysShort:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],daysMin:["Su","Mo","Tu","We","Th","Fr","Sa"],months:["January","February","March","April","May","June","July","August","September","October","November","December"],monthsShort:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],today:"Today",clear:"Clear",dateFormat:"mm/dd/yyyy",timeFormat:"hh:ii aa",firstDay:0}
})(jQuery);
(function(a){a.fn.datepicker.language.zh={days:["周日","周一","周二","周三","周四","周五","周六"],daysShort:["日","一","二","三","四","五","六"],daysMin:["日","一","二","三","四","五","六"],months:["一月","二月","三月","四月","五月","六月","七月","八月","九月","十月","十一月","十二月"],monthsShort:["一月","二月","三月","四月","五月","六月","七月","八月","九月","十月","十一月","十二月"],today:"今天",clear:"清除",dateFormat:"yyyy-mm-dd",timeFormat:"hh:ii",firstDay:1}
})(jQuery);
(function(a){a.fn.datepicker.language.ja={days:["日","月","火","水","木","金","土"],daysShort:["日","月","火","水","木","金","土"],daysMin:["日","月","火","水","木","金","土"],months:["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"],monthsShort:["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"],today:"今日",clear:"クリア",dateFormat:"yyyy-mm-dd",timeFormat:"hh:ii",firstDay:1}
})(jQuery);
(function(a){a.fn.datepicker.language.ko={days:["일","월","화","수","목","금","토"],daysShort:["일","월","화","수","목","금","토"],daysMin:["일","월","화","수","목","금","토"],months:["년 1월","년 2월","년 3월","년 4월","년 5월","년 6월","년 7월","년 8월","년 9월","년 10월","년 11월","년 12월"],monthsShort:["년 1월","년 2월","년 3월","년 4월","년 5월","년 6월","년 7월","년 8월","년 9월","년 10월","년 11월","년 12월"],today:"오늘",clear:"명확한",dateFormat:"yyyy-mm-dd",timeFormat:"hh:ii",firstDay:1}
})(jQuery);
(function(a){a.fn.mbPagination=function(e){var h={perPage:10,prevText:"&#10092;",nextText:"&#10093;",showNextPrev:false,showFirstLast:false,firstText:"&#10092;&#10092;",lastText:"&#10093;&#10093;",tabElemClick:""};
var o=a.extend({},h,e);
var s=o.perPage;
var q=this;
var p=this.attr("id");
if(o.tabElemClick!=""){var n=a("#"+o.tabElemClick+" #"+p).children(),m=a("#"+o.tabElemClick+" .number-li")
}else{var n=a(q).children(),m=a("#"+o.tabElemClick+" .number-li")
}var d=n.length;
for(var l=0;
l<d;
l++){if(l>(o.perPage-1)){n.eq(l).hide()
}}var u=Math.ceil(d/s);
if(d>s){var k='<div class="paging" id="paging-'+o.tabElemClick+'"><ul class="mb-pagination">';
var f=a("ul.mb-pagination > li.active");
var r=a(f).find("a").text();
if(o.showFirstLast){if(r=="1"){k+='<li class="first-list non-number-li" disabled><a href="javascript:;">'+o.firstText+"</a></li>"
}else{k+='<li class="first-list non-number-li"><a href="javascript:;">'+o.firstText+"</a></li>"
}}if(o.showNextPrev){if(r=="1"){k+='<li class="prev-list non-number-li" disabled><a href="javascript:;">'+o.prevText+"</a></li>"
}else{k+='<li class="prev-list non-number-li"><a href="javascript:;">'+o.prevText+"</a></li>"
}}for(var l=1;
l<=u;
l++){var t="";
if(l==1){t=" active"
}k+='<li class="number-li'+t+'"><a href="javascript:;" >'+l+"</a></li>"
}if(o.showNextPrev){if(r==u){k+='<li class="next-list non-number-li" disabled><a href="javascript:;">'+o.nextText+"</a></li>"
}else{k+='<li class="next-list non-number-li"><a href="javascript:;">'+o.nextText+"</a></li>"
}}if(o.showFirstLast){if(r==u){k+='<li class="last-list non-number-li"><a href="javascript:;">'+o.lastText+"</a></li>"
}else{k+='<li class="last-list non-number-li"><a href="javascript:;">'+o.lastText+"</a></li>"
}}k+="</ul></div>";
var g=a(this).parent().attr("id");
var j=a(this).parent().parent().attr("id");
if(o.tabElemClick!=""){if(o.tabElemClick==g||o.tabElemClick==j){a("#"+o.tabElemClick+" #"+p).after(k)
}else{a(this).after(k)
}}}m.first().addClass("active");
function b(){a("#"+o.tabElemClick+" .number-li:not(.active)").hide();
a(".dots").each(function(){a(this).remove(".dots")
});
if(a("#"+o.tabElemClick+" .number-li").hasClass("active")){var i=a("#"+o.tabElemClick+" .number-li.active");
i.show();
if(i.prev().html()!==a("#"+o.tabElemClick+" .number-li").first().html()){i.prev("#"+o.tabElemClick+" .number-li").show().before('<li class="dots"><span>...</span></li>')
}else{a("#"+o.tabElemClick+" .number-li").first().show()
}if(i.next().html()!==a("#"+o.tabElemClick+" .number-li").last().html()){i.next("#"+o.tabElemClick+" .number-li").show().after('<li class="dots"><span>...</span></li>')
}else{a("#"+o.tabElemClick+" .number-li").last().show()
}}}function c(){if(a("#"+o.tabElemClick+" .non-number-li").prev().hasClass("active")){a("#"+o.tabElemClick+" .next-list").addClass("disabled").next().addClass("disabled");
a("#"+o.tabElemClick+" .prev-list").removeClass("disabled").prev().removeClass("disabled")
}else{if(a("#"+o.tabElemClick+" .non-number-li").next().hasClass("active")){a("#"+o.tabElemClick+" .next-list").removeClass("disabled").next().removeClass("disabled");
a("#"+o.tabElemClick+" .prev-list").addClass("disabled").prev().addClass("disabled")
}else{a("#"+o.tabElemClick+" .non-number-li").each(function(){a(this).removeClass("disabled")
})
}}}c();
b();
a("body").on("click",".mb-pagination a",function(C){var E=a(this),A=a(this).parent(),G=a(this).parent().parent().parent().siblings("#"+p),H=a(this).parent().parent().parent().parent().attr("id"),D=a("#"+o.tabElemClick+" .number-li.active");
var z=a(G).children();
var v=z.length;
function i(K,J){for(var I=K;
I<J;
I++){z.eq(I).show()
}for(var I=0;
I<v;
I++){if((I<K)||(I>=J)){z.eq(I).hide()
}}}function F(I){I.addClass("active").siblings().removeClass("active")
}function y(){a("#"+o.tabElemClick+" .non-number-li").removeClass("disabled")
}if(A.hasClass("number-li")){F(E.parent());
i((E.text()-1)*s,E.text()*s)
}else{if(A.hasClass("prev-list")){if(D.prev().hasClass("non-number-li")){return false
}else{var x=(Number(D.prev(m).text())-1)*s,B=Number(D.prev(m).text())*s;
i(x,B);
F(D.prev())
}}else{if(A.hasClass("next-list")){if(D.next().hasClass("non-number-li")){return false
}else{var x=(Number(D.next(m).text())-1)*s,B=Number(D.next(m).text())*s;
i(x,B);
F(D.next())
}}}if(A.hasClass("last-list")){i((v-s),v);
F(a("#"+o.tabElemClick+" .number-li").eq(a("#"+o.tabElemClick+" .number-li").length-1))
}if(A.hasClass("first-list")){i(0,s);
F(a("#"+o.tabElemClick+" .number-li").eq(0))
}}b();
c();
var w=a("#"+H).offset().top;
w=Number(w)-100;
console.log(w);
a("html, body").animate({scrollTop:w},500);
return false
})
}
})(jQuery);
!function(a,b){"function"==typeof define&&define.amd?define(["jquery"],b):b(a.jQuery)
}(this,function(g){function j(a){if(p.webkit&&!a){return{height:0,width:0}
}if(!p.data.outer){var c={border:"none","box-sizing":"content-box",height:"200px",margin:"0",padding:"0",width:"200px"};
p.data.inner=g("<div>").css(g.extend({},c)),p.data.outer=g("<div>").css(g.extend({left:"-1000px",overflow:"scroll",position:"absolute",top:"-1000px"},c)).append(p.data.inner).appendTo("body")
}return p.data.outer.scrollLeft(1000).scrollTop(1000),{height:Math.ceil(p.data.outer.offset().top-p.data.inner.offset().top||0),width:Math.ceil(p.data.outer.offset().left-p.data.inner.offset().left||0)}
}function d(){var a=j(!0);
return !(a.height||a.width)
}function q(a){var c=a.originalEvent;
return c.axis&&c.axis===c.HORIZONTAL_AXIS?!1:c.wheelDeltaX?!1:!0
}var b=!1,p={data:{index:0,name:"scrollbar"},macosx:/mac/i.test(navigator.platform),mobile:/android|webos|iphone|ipad|ipod|blackberry/i.test(navigator.userAgent),overlay:null,scroll:null,scrolls:[],webkit:/webkit/i.test(navigator.userAgent)&&!/edge\/\d+/i.test(navigator.userAgent)};
p.scrolls.add=function(a){this.remove(a).push(a)
},p.scrolls.remove=function(a){for(;
g.inArray(a,this)>=0;
){this.splice(g.inArray(a,this),1)
}return this
};
var h={autoScrollSize:!0,autoUpdate:!0,debug:!1,disableBodyScroll:!1,duration:200,ignoreMobile:!1,ignoreOverlay:!1,scrollStep:30,showArrows:!1,stepScrolling:!0,scrollx:null,scrolly:null,onDestroy:null,onInit:null,onScroll:null,onUpdate:null},f=function(a){p.scroll||(p.overlay=d(),p.scroll=j(),m(),g(window).resize(function(){var c=!1;
if(p.scroll&&(p.scroll.height||p.scroll.width)){var e=j();
(e.height!==p.scroll.height||e.width!==p.scroll.width)&&(p.scroll=e,c=!0)
}m(c)
})),this.container=a,this.namespace=".scrollbar_"+p.data.index++,this.options=g.extend({},h,window.jQueryScrollbarOptions||{}),this.scrollTo=null,this.scrollx={},this.scrolly={},a.data(p.data.name,this),p.scrolls.add(this)
};
f.prototype={destroy:function(){if(this.wrapper){this.container.removeData(p.data.name),p.scrolls.remove(this);
var a=this.container.scrollLeft(),c=this.container.scrollTop();
this.container.insertBefore(this.wrapper).css({height:"",margin:"","max-height":""}).removeClass("scroll-content scroll-scrollx_visible scroll-scrolly_visible").off(this.namespace).scrollLeft(a).scrollTop(c),this.scrollx.scroll.removeClass("scroll-scrollx_visible").find("div").andSelf().off(this.namespace),this.scrolly.scroll.removeClass("scroll-scrolly_visible").find("div").andSelf().off(this.namespace),this.wrapper.remove(),g(document).add("body").off(this.namespace),g.isFunction(this.options.onDestroy)&&this.options.onDestroy.apply(this,[this.container])
}},init:function(x){var t=this,l=this.container,v=this.containerWrapper||l,u=this.namespace,z=g.extend(this.options,x||{}),A={x:this.scrollx,y:this.scrolly},y=this.wrapper,w={scrollLeft:l.scrollLeft(),scrollTop:l.scrollTop()};
if(p.mobile&&z.ignoreMobile||p.overlay&&z.ignoreOverlay||p.macosx&&!p.webkit){return !1
}if(y){v.css({height:"auto","margin-bottom":-1*p.scroll.height+"px","margin-right":-1*p.scroll.width+"px","max-height":""})
}else{if(this.wrapper=y=g("<div>").addClass("scroll-wrapper").addClass(l.attr("class")).css("position","absolute"==l.css("position")?"absolute":"relative").insertBefore(l).append(l),l.is("textarea")&&(this.containerWrapper=v=g("<div>").insertBefore(l).append(l),y.addClass("scroll-textarea")),v.addClass("scroll-content").css({height:"auto","margin-bottom":-1*p.scroll.height+"px","margin-right":-1*p.scroll.width+"px","max-height":""}),l.on("scroll"+u,function(a){g.isFunction(z.onScroll)&&z.onScroll.call(t,{maxScroll:A.y.maxScrollOffset,scroll:l.scrollTop(),size:A.y.size,visible:A.y.visible},{maxScroll:A.x.maxScrollOffset,scroll:l.scrollLeft(),size:A.x.size,visible:A.x.visible}),A.x.isVisible&&A.x.scroll.bar.css("left",l.scrollLeft()*A.x.kx+"px"),A.y.isVisible&&A.y.scroll.bar.css("top",l.scrollTop()*A.y.kx+"px")
}),y.on("scroll"+u,function(){y.scrollTop(0).scrollLeft(0)
}),z.disableBodyScroll){var s=function(a){q(a)?A.y.isVisible&&A.y.mousewheel(a):A.x.isVisible&&A.x.mousewheel(a)
};
y.on("MozMousePixelScroll"+u,s),y.on("mousewheel"+u,s),p.mobile&&y.on("touchstart"+u,function(i){var n=i.originalEvent.touches&&i.originalEvent.touches[0]||i,c={pageX:n.pageX,pageY:n.pageY},a={left:l.scrollLeft(),top:l.scrollTop()};
g(document).on("touchmove"+u,function(o){var r=o.originalEvent.targetTouches&&o.originalEvent.targetTouches[0]||o;
l.scrollLeft(a.left+c.pageX-r.pageX),l.scrollTop(a.top+c.pageY-r.pageY),o.preventDefault()
}),g(document).on("touchend"+u,function(){g(document).off(u)
})
})
}g.isFunction(z.onInit)&&z.onInit.apply(this,[l])
}g.each(A,function(C,n){var c=null,D=1,o="x"===C?"scrollLeft":"scrollTop",B=z.scrollStep,a=function(){var e=l[o]();
l[o](e+B),1==D&&e+B>=r&&(e=l[o]()),-1==D&&r>=e+B&&(e=l[o]()),l[o]()==e&&c&&c()
},r=0;
n.scroll||(n.scroll=t._getScroll(z["scroll"+C]).addClass("scroll-"+C),z.showArrows&&n.scroll.addClass("scroll-element_arrows_visible"),n.mousewheel=function(e){if(!n.isVisible||"x"===C&&q(e)){return !0
}if("y"===C&&!q(e)){return A.x.mousewheel(e),!0
}var E=-1*e.originalEvent.wheelDelta||e.originalEvent.detail,F=n.size-n.visible-n.offset;
return(E>0&&F>r||0>E&&r>0)&&(r+=E,0>r&&(r=0),r>F&&(r=F),t.scrollTo=t.scrollTo||{},t.scrollTo[o]=r,setTimeout(function(){t.scrollTo&&(l.stop().animate(t.scrollTo,240,"linear",function(){r=l[o]()
}),t.scrollTo=null)
},1)),e.preventDefault(),!1
},n.scroll.on("MozMousePixelScroll"+u,n.mousewheel).on("mousewheel"+u,n.mousewheel).on("mouseenter"+u,function(){r=l[o]()
}),n.scroll.find(".scroll-arrow, .scroll-element_track").on("mousedown"+u,function(E){if(1!=E.which){return !0
}D=1;
var F={eventOffset:E["x"===C?"pageX":"pageY"],maxScrollValue:n.size-n.visible-n.offset,scrollbarOffset:n.scroll.bar.offset()["x"===C?"left":"top"],scrollbarSize:n.scroll.bar["x"===C?"outerWidth":"outerHeight"]()},e=0,i=0;
return g(this).hasClass("scroll-arrow")?(D=g(this).hasClass("scroll-arrow_more")?1:-1,B=z.scrollStep*D,r=D>0?F.maxScrollValue:0):(D=F.eventOffset>F.scrollbarOffset+F.scrollbarSize?1:F.eventOffset<F.scrollbarOffset?-1:0,B=Math.round(0.75*n.visible)*D,r=F.eventOffset-F.scrollbarOffset-(z.stepScrolling?1==D?F.scrollbarSize:0:Math.round(F.scrollbarSize/2)),r=l[o]()+r/n.kx),t.scrollTo=t.scrollTo||{},t.scrollTo[o]=z.stepScrolling?l[o]()+B:r,z.stepScrolling&&(c=function(){r=l[o](),clearInterval(i),clearTimeout(e),e=0,i=0
},e=setTimeout(function(){i=setInterval(a,40)
},z.duration+100)),setTimeout(function(){t.scrollTo&&(l.animate(t.scrollTo,z.duration),t.scrollTo=null)
},1),t._handleMouseDown(c,E)
}),n.scroll.bar.on("mousedown"+u,function(E){if(1!=E.which){return !0
}var e=E["x"===C?"pageX":"pageY"],F=l[o]();
return n.scroll.addClass("scroll-draggable"),g(document).on("mousemove"+u,function(i){var G=parseInt((i["x"===C?"pageX":"pageY"]-e)/n.kx,10);
l[o](F+G)
}),t._handleMouseDown(function(){n.scroll.removeClass("scroll-draggable"),r=l[o]()
},E)
}))
}),g.each(A,function(a,i){var n="scroll-scroll"+a+"_visible",c="x"==a?A.y:A.x;
i.scroll.removeClass(n),c.scroll.removeClass(n),v.removeClass(n)
}),g.each(A,function(a,c){g.extend(c,"x"==a?{offset:parseInt(l.css("left"),10)||0,size:l.prop("scrollWidth"),visible:y.width()}:{offset:parseInt(l.css("top"),10)||0,size:l.prop("scrollHeight"),visible:y.height()})
}),this._updateScroll("x",this.scrollx),this._updateScroll("y",this.scrolly),g.isFunction(z.onUpdate)&&z.onUpdate.apply(this,[l]),g.each(A,function(B,D){var c="x"===B?"left":"top",H="x"===B?"outerWidth":"outerHeight",G="x"===B?"width":"height",C=parseInt(l.css(c),10)||0,r=D.size,F=D.visible+C,E=D.scroll.size[H]()+(parseInt(D.scroll.size.css(c),10)||0);
z.autoScrollSize&&(D.scrollbarSize=parseInt(E*F/r,10),D.scroll.bar.css(G,D.scrollbarSize+"px")),D.scrollbarSize=D.scroll.bar[H](),D.kx=(E-D.scrollbarSize)/(r-F)||1,D.maxScrollOffset=r-F
}),l.scrollLeft(w.scrollLeft).scrollTop(w.scrollTop).trigger("scroll")
},_getScroll:function(a){var c={advanced:['<div class="scroll-element">','<div class="scroll-element_corner"></div>','<div class="scroll-arrow scroll-arrow_less"></div>','<div class="scroll-arrow scroll-arrow_more"></div>','<div class="scroll-element_outer">','<div class="scroll-element_size"></div>','<div class="scroll-element_inner-wrapper">','<div class="scroll-element_inner scroll-element_track">','<div class="scroll-element_inner-bottom"></div>',"</div>","</div>",'<div class="scroll-bar">','<div class="scroll-bar_body">','<div class="scroll-bar_body-inner"></div>',"</div>",'<div class="scroll-bar_bottom"></div>','<div class="scroll-bar_center"></div>',"</div>","</div>","</div>"].join(""),simple:['<div class="scroll-element">','<div class="scroll-element_outer">','<div class="scroll-element_size"></div>','<div class="scroll-element_track"></div>','<div class="scroll-bar"></div>',"</div>","</div>"].join("")};
return c[a]&&(a=c[a]),a||(a=c.simple),a="string"==typeof a?g(a).appendTo(this.wrapper):g(a),g.extend(a,{bar:a.find(".scroll-bar"),size:a.find(".scroll-element_size"),track:a.find(".scroll-element_track")}),a
},_handleMouseDown:function(c,i){var a=this.namespace;
return g(document).on("blur"+a,function(){g(document).add("body").off(a),c&&c()
}),g(document).on("dragstart"+a,function(e){return e.preventDefault(),!1
}),g(document).on("mouseup"+a,function(){g(document).add("body").off(a),c&&c()
}),g("body").on("selectstart"+a,function(e){return e.preventDefault(),!1
}),i&&i.preventDefault(),!1
},_updateScroll:function(x,t){var B=this.container,l=this.containerWrapper||B,v="scroll-scroll"+x+"_visible",u="x"===x?this.scrolly:this.scrollx,z=parseInt(this.container.css("x"===x?"left":"top"),10)||0,A=this.wrapper,y=t.size,w=t.visible+z;
t.isVisible=y-w>1,t.isVisible?(t.scroll.addClass(v),u.scroll.addClass(v),l.addClass(v)):(t.scroll.removeClass(v),u.scroll.removeClass(v),l.removeClass(v)),"y"===x&&(B.is("textarea")||w>y?l.css({height:w+p.scroll.height+"px","max-height":"none"}):l.css({"max-height":w+p.scroll.height+"px"})),(t.size!=B.prop("scrollWidth")||u.size!=B.prop("scrollHeight")||t.visible!=A.width()||u.visible!=A.height()||t.offset!=(parseInt(B.css("left"),10)||0)||u.offset!=(parseInt(B.css("top"),10)||0))&&(g.extend(this.scrollx,{offset:parseInt(B.css("left"),10)||0,size:B.prop("scrollWidth"),visible:A.width()}),g.extend(this.scrolly,{offset:parseInt(B.css("top"),10)||0,size:this.container.prop("scrollHeight"),visible:A.height()}),this._updateScroll("x"===x?"y":"x",u))
}};
var k=f;
g.fn.scrollbar=function(a,c){return"string"!=typeof a&&(c=a,a="init"),"undefined"==typeof c&&(c=[]),g.isArray(c)||(c=[c]),this.not("body, .scroll-wrapper").each(function(){var e=g(this),i=e.data(p.data.name);
(i||"init"===a)&&(i||(i=new k(e)),i[a]&&i[a].apply(i,c))
}),this
},g.fn.scrollbar.options=h;
var m=function(){var a=0,c=0;
return function(v){var r,e,x,w,u,l,t;
for(r=0;
r<p.scrolls.length;
r++){w=p.scrolls[r],e=w.container,x=w.options,u=w.wrapper,l=w.scrollx,t=w.scrolly,(v||x.autoUpdate&&u&&u.is(":visible")&&(e.prop("scrollWidth")!=l.size||e.prop("scrollHeight")!=t.size||u.width()!=l.visible||u.height()!=t.visible))&&(w.init(),x.debug&&(window.console&&console.log({scrollHeight:e.prop("scrollHeight")+":"+w.scrolly.size,scrollWidth:e.prop("scrollWidth")+":"+w.scrollx.size,visibleHeight:u.height()+":"+w.scrolly.visible,visibleWidth:u.width()+":"+w.scrollx.visible},!0),c++))
}b&&c>10?(window.console&&console.log("Scroll updates exceed 10"),m=function(){}):(clearTimeout(a),a=setTimeout(m,300))
}
}();
window.angular&&!function(a){a.module("jQueryScrollbar",[]).provider("jQueryScrollbar",function(){var c=h;
return{setOptions:function(e){a.extend(c,e)
},$get:function(){return{options:a.copy(c)}
}}
}).directive("jqueryScrollbar",["jQueryScrollbar","$parse",function(c,i){return{restrict:"AC",link:function(v,n,u){var l=i(u.jqueryScrollbar),e=l(v);
n.scrollbar(e||c.options).on("$destroy",function(){n.scrollbar("destroy")
})
}}
}])
}(window.angular)
});
/*! jQuery Validation Plugin - v1.17.0 - 7/29/2017
 * https://jqueryvalidation.org/
 * Copyright (c) 2017 Jörn Zaefferer; Licensed MIT */
;
!function(b){"function"==typeof define&&define.amd?define(["jquery"],b):"object"==typeof module&&module.exports?module.exports=b(require("jquery")):b(jQuery)
}(function(e){e.extend(e.fn,{validate:function(a){if(!this.length){return void (a&&a.debug&&window.console&&console.warn("Nothing selected, can't validate, returning nothing."))
}var g=e.data(this[0],"validator");
return g?g:(this.attr("novalidate","novalidate"),g=new e.validator(a,this[0]),e.data(this[0],"validator",g),g.settings.onsubmit&&(this.on("click.validate",":submit",function(c){g.submitButton=c.currentTarget,e(this).hasClass("cancel")&&(g.cancelSubmit=!0),void 0!==e(this).attr("formnovalidate")&&(g.cancelSubmit=!0)
}),this.on("submit.validate",function(c){function h(){var i,b;
return g.submitButton&&(g.settings.submitHandler||g.formSubmitted)&&(i=e("<input type='hidden'/>").attr("name",g.submitButton.name).val(e(g.submitButton).val()).appendTo(g.currentForm)),!g.settings.submitHandler||(b=g.settings.submitHandler.call(g,g.currentForm,c),i&&i.remove(),void 0!==b&&b)
}return g.settings.debug&&c.preventDefault(),g.cancelSubmit?(g.cancelSubmit=!1,h()):g.form()?g.pendingRequest?(g.formSubmitted=!0,!1):h():(g.focusInvalid(),!1)
})),g)
},valid:function(){var a,h,g;
return e(this[0]).is("form")?a=this.validate().form():(g=[],a=!0,h=e(this[0].form).validate(),this.each(function(){a=h.element(this)&&a,a||(g=g.concat(h.errorList))
}),h.errorList=g),a
},rules:function(r,q){var p,o,n,m,l,k,a=this[0];
if(null!=a&&(!a.form&&a.hasAttribute("contenteditable")&&(a.form=this.closest("form")[0],a.name=this.attr("name")),null!=a.form)){if(r){switch(p=e.data(a.form,"validator").settings,o=p.rules,n=e.validator.staticRules(a),r){case"add":e.extend(n,e.validator.normalizeRule(q)),delete n.messages,o[a.name]=n,q.messages&&(p.messages[a.name]=e.extend(p.messages[a.name],q.messages));
break;
case"remove":return q?(k={},e.each(q.split(/\s/),function(g,c){k[c]=n[c],delete n[c]
}),k):(delete o[a.name],n)
}}return m=e.validator.normalizeRules(e.extend({},e.validator.classRules(a),e.validator.attributeRules(a),e.validator.dataRules(a),e.validator.staticRules(a)),a),m.required&&(l=m.required,delete m.required,m=e.extend({required:l},m)),m.remote&&(l=m.remote,delete m.remote,m=e.extend(m,{remote:l})),m
}}}),e.extend(e.expr.pseudos||e.expr[":"],{blank:function(a){return !e.trim(""+e(a).val())
},filled:function(a){var g=e(a).val();
return null!==g&&!!e.trim(""+g)
},unchecked:function(a){return !e(a).prop("checked")
}}),e.validator=function(a,g){this.settings=e.extend(!0,{},e.validator.defaults,a),this.currentForm=g,this.init()
},e.validator.format=function(a,g){return 1===arguments.length?function(){var b=e.makeArray(arguments);
return b.unshift(a),e.validator.format.apply(this,b)
}:void 0===g?a:(arguments.length>2&&g.constructor!==Array&&(g=e.makeArray(arguments).slice(1)),g.constructor!==Array&&(g=[g]),e.each(g,function(b,h){a=a.replace(new RegExp("\\{"+b+"\\}","g"),function(){return h
})
}),a)
},e.extend(e.validator,{defaults:{messages:{},groups:{},rules:{},errorClass:"error",pendingClass:"pending",validClass:"valid",errorElement:"label",focusCleanup:!1,focusInvalid:!0,errorContainer:e([]),errorLabelContainer:e([]),onsubmit:!0,ignore:":hidden",ignoreTitle:!1,onfocusin:function(b){this.lastActive=b,this.settings.focusCleanup&&(this.settings.unhighlight&&this.settings.unhighlight.call(this,b,this.settings.errorClass,this.settings.validClass),this.hideThese(this.errorsFor(b)))
},onfocusout:function(b){this.checkable(b)||!(b.name in this.submitted)&&this.optional(b)||this.element(b)
},onkeyup:function(a,h){var g=[16,17,18,20,35,36,37,38,39,40,45,144,225];
9===h.which&&""===this.elementValue(a)||e.inArray(h.keyCode,g)!==-1||(a.name in this.submitted||a.name in this.invalid)&&this.element(a)
},onclick:function(b){b.name in this.submitted?this.element(b):b.parentNode.name in this.submitted&&this.element(b.parentNode)
},highlight:function(a,h,g){"radio"===a.type?this.findByName(a.name).addClass(h).removeClass(g):e(a).addClass(h).removeClass(g)
},unhighlight:function(a,h,g){"radio"===a.type?this.findByName(a.name).removeClass(h).addClass(g):e(a).removeClass(h).addClass(g)
}},setDefaults:function(a){e.extend(e.validator.defaults,a)
},messages:{required:"This field is required.",remote:"Please fix this field.",email:"Please enter a valid email address.",url:"Please enter a valid URL.",date:"Please enter a valid date.",dateISO:"Please enter a valid date (ISO).",number:"Please enter a valid number.",digits:"Please enter only digits.",equalTo:"Please enter the same value again.",maxlength:e.validator.format("Please enter no more than {0} characters."),minlength:e.validator.format("Please enter at least {0} characters."),rangelength:e.validator.format("Please enter a value between {0} and {1} characters long."),range:e.validator.format("Please enter a value between {0} and {1}."),max:e.validator.format("Please enter a value less than or equal to {0}."),min:e.validator.format("Please enter a value greater than or equal to {0}."),step:e.validator.format("Please enter a multiple of {0}.")},autoCreateRanges:!1,prototype:{init:function(){function a(i){!this.form&&this.hasAttribute("contenteditable")&&(this.form=e(this).closest("form")[0],this.name=e(this).attr("name"));
var l=e.data(this.form,"validator"),k="on"+i.type.replace(/^validate/,""),j=l.settings;
j[k]&&!e(this).is(j.ignore)&&j[k].call(l,this,i)
}this.labelContainer=e(this.settings.errorLabelContainer),this.errorContext=this.labelContainer.length&&this.labelContainer||e(this.currentForm),this.containers=e(this.settings.errorContainer).add(this.settings.errorLabelContainer),this.submitted={},this.valueCache={},this.pendingRequest=0,this.pending={},this.invalid={},this.reset();
var h,g=this.groups={};
e.each(this.settings.groups,function(i,j){"string"==typeof j&&(j=j.split(/\s/)),e.each(j,function(b,k){g[k]=i
})
}),h=this.settings.rules,e.each(h,function(c,i){h[c]=e.validator.normalizeRule(i)
}),e(this.currentForm).on("focusin.validate focusout.validate keyup.validate",":text, [type='password'], [type='file'], select, textarea, [type='number'], [type='search'], [type='tel'], [type='url'], [type='email'], [type='datetime'], [type='date'], [type='month'], [type='week'], [type='time'], [type='datetime-local'], [type='range'], [type='color'], [type='radio'], [type='checkbox'], [contenteditable], [type='button']",a).on("click.validate","select, option, [type='radio'], [type='checkbox']",a),this.settings.invalidHandler&&e(this.currentForm).on("invalid-form.validate",this.settings.invalidHandler)
},form:function(){return this.checkForm(),e.extend(this.submitted,this.errorMap),this.invalid=e.extend({},this.errorMap),this.valid()||e(this.currentForm).triggerHandler("invalid-form",[this]),this.showErrors(),this.valid()
},checkForm:function(){this.prepareForm();
for(var g=0,c=this.currentElements=this.elements();
c[g];
g++){this.check(c[g])
}return this.valid()
},element:function(a){var n,m,l=this.clean(a),k=this.validationTargetFor(l),j=this,i=!0;
return void 0===k?delete this.invalid[l.name]:(this.prepareElement(k),this.currentElements=e(k),m=this.groups[k.name],m&&e.each(this.groups,function(g,c){c===m&&g!==k.name&&(l=j.validationTargetFor(j.clean(j.findByName(g))),l&&l.name in j.invalid&&(j.currentElements.push(l),i=j.check(l)&&i))
}),n=this.check(k)!==!1,i=i&&n,n?this.invalid[k.name]=!1:this.invalid[k.name]=!0,this.numberOfInvalids()||(this.toHide=this.toHide.add(this.containers)),this.showErrors(),e(a).attr("aria-invalid",!n)),i
},showErrors:function(a){if(a){var g=this;
e.extend(this.errorMap,a),this.errorList=e.map(this.errorMap,function(h,c){return{message:h,element:g.findByName(c)[0]}
}),this.successList=e.grep(this.successList,function(b){return !(b.name in a)
})
}this.settings.showErrors?this.settings.showErrors.call(this,this.errorMap,this.errorList):this.defaultShowErrors()
},resetForm:function(){e.fn.resetForm&&e(this.currentForm).resetForm(),this.invalid={},this.submitted={},this.prepareForm(),this.hideErrors();
var a=this.elements().removeData("previousValue").removeAttr("aria-invalid");
this.resetElements(a)
},resetElements:function(g){var c;
if(this.settings.unhighlight){for(c=0;
g[c];
c++){this.settings.unhighlight.call(this,g[c],this.settings.errorClass,""),this.findByName(g[c].name).removeClass(this.settings.validClass)
}}else{g.removeClass(this.settings.errorClass).removeClass(this.settings.validClass)
}},numberOfInvalids:function(){return this.objectLength(this.invalid)
},objectLength:function(h){var g,i=0;
for(g in h){void 0!==h[g]&&null!==h[g]&&h[g]!==!1&&i++
}return i
},hideErrors:function(){this.hideThese(this.toHide)
},hideThese:function(b){b.not(this.containers).text(""),this.addWrapper(b).hide()
},valid:function(){return 0===this.size()
},size:function(){return this.errorList.length
},focusInvalid:function(){if(this.settings.focusInvalid){try{e(this.findLastActive()||this.errorList.length&&this.errorList[0].element||[]).filter(":visible").focus().trigger("focusin")
}catch(a){}}},findLastActive:function(){var a=this.lastActive;
return a&&1===e.grep(this.errorList,function(b){return b.element.name===a.name
}).length&&a
},elements:function(){var a=this,g={};
return e(this.currentForm).find("input, select, textarea, [contenteditable]").not(":submit, :reset, :image, :disabled").not(this.settings.ignore).filter(function(){var b=this.name||e(this).attr("name");
return !b&&a.settings.debug&&window.console&&console.error("%o has no name assigned",this),this.hasAttribute("contenteditable")&&(this.form=e(this).closest("form")[0],this.name=b),!(b in g||!a.objectLength(e(this).rules()))&&(g[b]=!0,!0)
})
},clean:function(a){return e(a)[0]
},errors:function(){var a=this.settings.errorClass.split(" ").join(".");
return e(this.settings.errorElement+"."+a,this.errorContext)
},resetInternals:function(){this.successList=[],this.errorList=[],this.errorMap={},this.toShow=e([]),this.toHide=e([])
},reset:function(){this.resetInternals(),this.currentElements=e([])
},prepareForm:function(){this.reset(),this.toHide=this.errors().add(this.containers)
},prepareElement:function(b){this.reset(),this.toHide=this.errorsFor(b)
},elementValue:function(a){var j,i,h=e(a),g=a.type;
return"radio"===g||"checkbox"===g?this.findByName(a.name).filter(":checked").val():"number"===g&&"undefined"!=typeof a.validity?a.validity.badInput?"NaN":h.val():(j=a.hasAttribute("contenteditable")?h.text():h.val(),"file"===g?"C:\\fakepath\\"===j.substr(0,12)?j.substr(12):(i=j.lastIndexOf("/"),i>=0?j.substr(i+1):(i=j.lastIndexOf("\\"),i>=0?j.substr(i+1):j)):"string"==typeof j?j.replace(/\r/g,""):j)
},check:function(t){t=this.validationTargetFor(this.clean(t));
var s,r,q,p,o=e(t).rules(),n=e.map(o,function(g,c){return c
}).length,m=!1,l=this.elementValue(t);
if("function"==typeof o.normalizer?p=o.normalizer:"function"==typeof this.settings.normalizer&&(p=this.settings.normalizer),p){if(l=p.call(t,l),"string"!=typeof l){throw new TypeError("The normalizer should return a string value.")
}delete o.normalizer
}for(r in o){q={method:r,parameters:o[r]};
try{if(s=e.validator.methods[r].call(this,l,t,q.parameters),"dependency-mismatch"===s&&1===n){m=!0;
continue
}if(m=!1,"pending"===s){return void (this.toHide=this.toHide.not(this.errorsFor(t)))
}if(!s){return this.formatAndAdd(t,q),!1
}}catch(a){throw this.settings.debug&&window.console&&console.log("Exception occurred when checking element "+t.id+", check the '"+q.method+"' method.",a),a instanceof TypeError&&(a.message+=".  Exception occurred when checking element "+t.id+", check the '"+q.method+"' method."),a
}}if(!m){return this.objectLength(o)&&this.successList.push(t),!0
}},customDataMessage:function(a,g){return e(a).data("msg"+g.charAt(0).toUpperCase()+g.substring(1).toLowerCase())||e(a).data("msg")
},customMessage:function(h,g){var i=this.settings.messages[h];
return i&&(i.constructor===String?i:i[g])
},findDefined:function(){for(var b=0;
b<arguments.length;
b++){if(void 0!==arguments[b]){return arguments[b]
}}},defaultMessage:function(a,i){"string"==typeof i&&(i={method:i});
var h=this.findDefined(this.customMessage(a.name,i.method),this.customDataMessage(a,i.method),!this.settings.ignoreTitle&&a.title||void 0,e.validator.messages[i.method],"<strong>Warning: No message defined for "+a.name+"</strong>"),g=/\$?\{(\d+)\}/g;
return"function"==typeof h?h=h.call(this,i.parameters,a):g.test(h)&&(h=e.validator.format(h.replace(g,"{$1}"),i.parameters)),h
},formatAndAdd:function(h,g){var i=this.defaultMessage(h,g);
this.errorList.push({message:i,element:h,method:g.method}),this.errorMap[h.name]=i,this.submitted[h.name]=i
},addWrapper:function(b){return this.settings.wrapper&&(b=b.add(b.parent(this.settings.wrapper))),b
},defaultShowErrors:function(){var h,g,i;
for(h=0;
this.errorList[h];
h++){i=this.errorList[h],this.settings.highlight&&this.settings.highlight.call(this,i.element,this.settings.errorClass,this.settings.validClass),this.showLabel(i.element,i.message)
}if(this.errorList.length&&(this.toShow=this.toShow.add(this.containers)),this.settings.success){for(h=0;
this.successList[h];
h++){this.showLabel(this.successList[h])
}}if(this.settings.unhighlight){for(h=0,g=this.validElements();
g[h];
h++){this.settings.unhighlight.call(this,g[h],this.settings.errorClass,this.settings.validClass)
}}this.toHide=this.toHide.not(this.toShow),this.hideErrors(),this.addWrapper(this.toShow).show()
},validElements:function(){return this.currentElements.not(this.invalidElements())
},invalidElements:function(){return e(this.errorList).map(function(){return this.element
})
},showLabel:function(r,q){var p,o,n,m,l=this.errorsFor(r),k=this.idOrName(r),a=e(r).attr("aria-describedby");
l.length?(l.removeClass(this.settings.validClass).addClass(this.settings.errorClass),l.html(q)):(l=e("<"+this.settings.errorElement+">").attr("id",k+"-error").addClass(this.settings.errorClass).html(q||""),p=l,this.settings.wrapper&&(p=l.hide().show().wrap("<"+this.settings.wrapper+"/>").parent()),this.labelContainer.length?this.labelContainer.append(p):this.settings.errorPlacement?this.settings.errorPlacement.call(this,p,e(r)):p.insertAfter(r),l.is("label")?l.attr("for",k):0===l.parents("label[for='"+this.escapeCssMeta(k)+"']").length&&(n=l.attr("id"),a?a.match(new RegExp("\\b"+this.escapeCssMeta(n)+"\\b"))||(a+=" "+n):a=n,e(r).attr("aria-describedby",a),o=this.groups[r.name],o&&(m=this,e.each(m.groups,function(g,h){h===o&&e("[name='"+m.escapeCssMeta(g)+"']",m.currentForm).attr("aria-describedby",l.attr("id"))
})))),!q&&this.settings.success&&(l.text(""),"string"==typeof this.settings.success?l.addClass(this.settings.success):this.settings.success(l,r)),this.toShow=this.toShow.add(l)
},errorsFor:function(a){var i=this.escapeCssMeta(this.idOrName(a)),h=e(a).attr("aria-describedby"),g="label[for='"+i+"'], label[for='"+i+"'] *";
return h&&(g=g+", #"+this.escapeCssMeta(h).replace(/\s+/g,", #")),this.errors().filter(g)
},escapeCssMeta:function(b){return b.replace(/([\\!"#$%&'()*+,.\/:;<=>?@\[\]^`{|}~])/g,"\\$1")
},idOrName:function(b){return this.groups[b.name]||(this.checkable(b)?b.name:b.id||b.name)
},validationTargetFor:function(a){return this.checkable(a)&&(a=this.findByName(a.name)),e(a).not(this.settings.ignore)[0]
},checkable:function(b){return/radio|checkbox/i.test(b.type)
},findByName:function(a){return e(this.currentForm).find("[name='"+this.escapeCssMeta(a)+"']")
},getLength:function(a,g){switch(g.nodeName.toLowerCase()){case"select":return e("option:selected",g).length;
case"input":if(this.checkable(g)){return this.findByName(g.name).filter(":checked").length
}}return a.length
},depend:function(g,c){return !this.dependTypes[typeof g]||this.dependTypes[typeof g](g,c)
},dependTypes:{"boolean":function(b){return b
},string:function(a,g){return !!e(a,g.form).length
},"function":function(g,c){return g(c)
}},optional:function(a){var g=this.elementValue(a);
return !e.validator.methods.required.call(this,g,a)&&"dependency-mismatch"
},startRequest:function(a){this.pending[a.name]||(this.pendingRequest++,e(a).addClass(this.settings.pendingClass),this.pending[a.name]=!0)
},stopRequest:function(a,g){this.pendingRequest--,this.pendingRequest<0&&(this.pendingRequest=0),delete this.pending[a.name],e(a).removeClass(this.settings.pendingClass),g&&0===this.pendingRequest&&this.formSubmitted&&this.form()?(e(this.currentForm).submit(),this.submitButton&&e("input:hidden[name='"+this.submitButton.name+"']",this.currentForm).remove(),this.formSubmitted=!1):!g&&0===this.pendingRequest&&this.formSubmitted&&(e(this.currentForm).triggerHandler("invalid-form",[this]),this.formSubmitted=!1)
},previousValue:function(a,g){return g="string"==typeof g&&g||"remote",e.data(a,"previousValue")||e.data(a,"previousValue",{old:null,valid:!0,message:this.defaultMessage(a,{method:g})})
},destroy:function(){this.resetForm(),e(this.currentForm).off(".validate").removeData("validator").find(".validate-equalTo-blur").off(".validate-equalTo").removeClass("validate-equalTo-blur")
}},classRuleSettings:{required:{required:!0},email:{email:!0},url:{url:!0},date:{date:!0},dateISO:{dateISO:!0},number:{number:!0},digits:{digits:!0},creditcard:{creditcard:!0}},addClassRules:function(a,g){a.constructor===String?this.classRuleSettings[a]=g:e.extend(this.classRuleSettings,a)
},classRules:function(a){var h={},g=e(a).attr("class");
return g&&e.each(g.split(" "),function(){this in e.validator.classRuleSettings&&e.extend(h,e.validator.classRuleSettings[this])
}),h
},normalizeAttributeRule:function(h,g,j,i){/min|max|step/.test(j)&&(null===g||/number|range|text/.test(g))&&(i=Number(i),isNaN(i)&&(i=void 0)),i||0===i?h[j]=i:g===j&&"range"!==g&&(h[j]=!0)
},attributeRules:function(a){var l,k,j={},i=e(a),h=a.getAttribute("type");
for(l in e.validator.methods){"required"===l?(k=a.getAttribute(l),""===k&&(k=!0),k=!!k):k=i.attr(l),this.normalizeAttributeRule(j,h,l,k)
}return j.maxlength&&/-1|2147483647|524288/.test(j.maxlength)&&delete j.maxlength,j
},dataRules:function(a){var l,k,j={},i=e(a),h=a.getAttribute("type");
for(l in e.validator.methods){k=i.data("rule"+l.charAt(0).toUpperCase()+l.substring(1).toLowerCase()),this.normalizeAttributeRule(j,h,l,k)
}return j
},staticRules:function(a){var h={},g=e.data(a.form,"validator");
return g.settings.rules&&(h=e.validator.normalizeRule(g.settings.rules[a.name])||{}),h
},normalizeRules:function(a,g){return e.each(a,function(h,c){if(c===!1){return void delete a[h]
}if(c.param||c.depends){var b=!0;
switch(typeof c.depends){case"string":b=!!e(c.depends,g.form).length;
break;
case"function":b=c.depends.call(g,g)
}b?a[h]=void 0===c.param||c.param:(e.data(g.form,"validator").resetElements(e(g)),delete a[h])
}}),e.each(a,function(c,b){a[c]=e.isFunction(b)&&"normalizer"!==c?b(g):b
}),e.each(["minlength","maxlength"],function(){a[this]&&(a[this]=Number(a[this]))
}),e.each(["rangelength","range"],function(){var b;
a[this]&&(e.isArray(a[this])?a[this]=[Number(a[this][0]),Number(a[this][1])]:"string"==typeof a[this]&&(b=a[this].replace(/[\[\]]/g,"").split(/[\s,]+/),a[this]=[Number(b[0]),Number(b[1])]))
}),e.validator.autoCreateRanges&&(null!=a.min&&null!=a.max&&(a.range=[a.min,a.max],delete a.min,delete a.max),null!=a.minlength&&null!=a.maxlength&&(a.rangelength=[a.minlength,a.maxlength],delete a.minlength,delete a.maxlength)),a
},normalizeRule:function(a){if("string"==typeof a){var g={};
e.each(a.split(/\s/),function(){g[this]=!0
}),a=g
}return a
},addMethod:function(a,h,g){e.validator.methods[a]=h,e.validator.messages[a]=void 0!==g?g:e.validator.messages[a],h.length<3&&e.validator.addClassRules(a,e.validator.normalizeRule(a))
},methods:{required:function(a,i,h){if(!this.depend(h,i)){return"dependency-mismatch"
}if("select"===i.nodeName.toLowerCase()){var g=e(i).val();
return g&&g.length>0
}return this.checkable(i)?this.getLength(a,i)>0:a.length>0
},email:function(g,c){return this.optional(c)||/^[a-z0-9]+([-._][a-z0-9]+)*@([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,4}$/.test(g)&&/^(?=.{1,64}@.{4,64}$)(?=.{6,100}$).*/.test(g)
},url:function(g,c){return this.optional(c)||/^(?:(?:(?:https?|ftp):)?\/\/)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})).?)(?::\d{2,5})?(?:[\/?#]\S*)?$/i.test(g)
},date:function(g,c){return this.optional(c)||!/Invalid|NaN/.test(new Date(g).toString())
},dateISO:function(g,c){return this.optional(c)||/^\d{4}[\/\-](0?[1-9]|1[012])[\/\-](0?[1-9]|[12][0-9]|3[01])$/.test(g)
},number:function(g,c){return this.optional(c)||/^(?:-?\d+|-?\d{1,3}(?:,\d{3})+)?(?:\.\d+)?$/.test(g)
},digits:function(g,c){return this.optional(c)||/^\d+$/.test(g)
},minlength:function(a,i,h){var g=e.isArray(a)?a.length:this.getLength(a,i);
return this.optional(i)||g>=h
},maxlength:function(a,i,h){var g=e.isArray(a)?a.length:this.getLength(a,i);
return this.optional(i)||g<=h
},rangelength:function(a,i,h){var g=e.isArray(a)?a.length:this.getLength(a,i);
return this.optional(i)||g>=h[0]&&g<=h[1]
},min:function(h,g,i){return this.optional(g)||h>=i
},max:function(h,g,i){return this.optional(g)||h<=i
},range:function(h,g,i){return this.optional(g)||h>=i[0]&&h<=i[1]
},step:function(x,w,v){var u,t=e(w).attr("type"),s="Step attribute on input type "+t+" is not supported.",r=["text","number","range"],q=new RegExp("\\b"+t+"\\b"),p=t&&!q.test(r.join()),o=function(g){var c=(""+g).match(/(?:\.(\d+))?$/);
return c&&c[1]?c[1].length:0
},n=function(b){return Math.round(b*Math.pow(10,u))
},a=!0;
if(p){throw new Error(s)
}return u=o(v),(o(x)>u||n(x)%n(v)!==0)&&(a=!1),this.optional(w)||a
},equalTo:function(a,i,h){var g=e(h);
return this.settings.onfocusout&&g.not(".validate-equalTo-blur").length&&g.addClass("validate-equalTo-blur").on("blur.validate-equalTo",function(){e(i).valid()
}),a===g.val()
},remote:function(a,p,o,n){if(this.optional(p)){return"dependency-mismatch"
}n="string"==typeof n&&n||"remote";
var m,l,k,j=this.previousValue(p,n);
return this.settings.messages[p.name]||(this.settings.messages[p.name]={}),j.originalMessage=j.originalMessage||this.settings.messages[p.name][n],this.settings.messages[p.name][n]=j.message,o="string"==typeof o&&{url:o}||o,k=e.param(e.extend({data:a},o.data)),j.old===k?j.valid:(j.old=k,m=this,this.startRequest(p),l={},l[p.name]=a,e.ajax(e.extend(!0,{mode:"abort",port:"validate"+p.name,dataType:"json",data:l,context:m.currentForm,success:function(b){var r,q,i,c=b===!0||"true"===b;
m.settings.messages[p.name][n]=j.originalMessage,c?(i=m.formSubmitted,m.resetInternals(),m.toHide=m.errorsFor(p),m.formSubmitted=i,m.successList.push(p),m.invalid[p.name]=!1,m.showErrors()):(r={},q=b||m.defaultMessage(p,{method:n,parameters:a}),r[p.name]=j.message=q,m.invalid[p.name]=!0,m.showErrors(r)),j.valid=c,m.stopRequest(p,c)
}},o)),"pending")
}}});
var d,f={};
return e.ajaxPrefilter?e.ajaxPrefilter(function(g,c,i){var h=g.port;
"abort"===g.mode&&(f[h]&&f[h].abort(),f[h]=i)
}):(d=e.ajax,e.ajax=function(c){var b=("mode" in c?c:e.ajaxSettings).mode,a=("port" in c?c:e.ajaxSettings).port;
return"abort"===b?(f[a]&&f[a].abort(),f[a]=d.apply(this,arguments),f[a]):d.apply(this,arguments)
}),e
});
if(!window.requestAnimationFrame){window.requestAnimationFrame=(function(){return window.webkitRequestAnimationFrame||window.mozRequestAnimationFrame||window.oRequestAnimationFrame||window.msRequestAnimationFrame||function(b,a){window.setTimeout(b,1000/60)
}
})()
}$("#form-register").validate({rules:{name:{required:true},radiogender:{required:true},nationality:{required:true},email:{required:true,email:true,minlength:6,maxlength:50},interest:{required:true},passwordnew:{required:true,minlength:8},passwordc:{required:true,equalTo:"#passwordnew",minlength:8}},messages:{name:{required:"This field is required, please fill it with your full name!"},passwordc:{equalTo:"This field is must be same with password!"}},submitHandler:function(a){alert("ok")
}});
$("#form-login").validate({rules:{emaillogin:{required:true,email:true,minlength:6,maxlength:50},passwordlogin:{required:true,minlength:8}},messages:{emaillogin:{required:"This field is required, please fill it with your correct username!"},passwordlogin:{required:"This field is required, please fill it with your correct password!"}},submitHandler:function(a){alert("ok")
}});
if(typeof imgUrl!=="undefined"){media360.start(imgUrl)
}if(typeof vid!=="undefined"){media360.startVideo()
}function getCookie(b){var e=b+"=";
var a=document.cookie.split(";");
for(var d=0;
d<a.length;
d++){var f=a[d];
while(f.charAt(0)==" "){f=f.substring(1,f.length)
}if(f.indexOf(e)==0){return f.substring(e.length,f.length)
}}return null
}function setCookie(b,f,c,e,a){console.log(b+"-"+f);
var g=new Date();
g.setDate(g.getDate()+c);
var d=escape(f)+((c==null)?"":"; expires="+g.toUTCString());
console.log(d);
if(window.location.host.indexOf("indonesia.travel")>-1){document.cookie=b+"="+f+";domain=.indonesia.travel;path=/"
}else{document.cookie=b+"="+f+";path=/"
}}$(document).ready(function(){if(window.location.href.indexOf("/sa/ar")>-1||window.location.href.indexOf("/uae/ar")>-1){setTimeout(function(){$("#datepickers-container > div").hide()
},300)
}});
+function(c){var d=function(g,f){this.$element=c(g);
this.$indicators=this.$element.find(".carousel-indicators");
this.options=f;
this.paused=null;
this.sliding=null;
this.interval=null;
this.$active=null;
this.$items=null;
this.options.keyboard&&this.$element.on("keydown.bs.carousel",c.proxy(this.keydown,this));
this.options.pause=="hover"&&!("ontouchstart" in document.documentElement)&&this.$element.on("mouseenter.bs.carousel",c.proxy(this.pause,this)).on("mouseleave.bs.carousel",c.proxy(this.cycle,this))
};
d.VERSION="3.3.5";
d.TRANSITION_DURATION=600;
d.DEFAULTS={interval:5000,pause:"hover",wrap:true,keyboard:true};
d.prototype.keydown=function(f){if(/input|textarea/i.test(f.target.tagName)){return
}switch(f.which){case 37:this.prev();
break;
case 39:this.next();
break;
default:return
}f.preventDefault()
};
d.prototype.cycle=function(f){f||(this.paused=false);
this.interval&&clearInterval(this.interval);
this.options.interval&&!this.paused&&(this.interval=setInterval(c.proxy(this.next,this),this.options.interval));
return this
};
d.prototype.getItemIndex=function(f){this.$items=f.parent().children(".item");
return this.$items.index(f||this.$active)
};
d.prototype.getItemForDirection=function(j,i){var f=this.getItemIndex(i);
var g=(j=="prev"&&f===0)||(j=="next"&&f==(this.$items.length-1));
if(g&&!this.options.wrap){return i
}var k=j=="prev"?-1:1;
var h=(f+k)%this.$items.length;
return this.$items.eq(h)
};
d.prototype.to=function(h){var g=this;
var f=this.getItemIndex(this.$active=this.$element.find(".item.active"));
if(h>(this.$items.length-1)||h<0){return
}if(this.sliding){return this.$element.one("slid.bs.carousel",function(){g.to(h)
})
}if(f==h){return this.pause().cycle()
}return this.slide(h>f?"next":"prev",this.$items.eq(h))
};
d.prototype.pause=function(f){f||(this.paused=true);
if(this.$element.find(".next, .prev").length&&c.support.transition){this.$element.trigger(c.support.transition.end);
this.cycle(true)
}this.interval=clearInterval(this.interval);
return this
};
d.prototype.next=function(){if(this.sliding){return
}return this.slide("next")
};
d.prototype.prev=function(){if(this.sliding){return
}return this.slide("prev")
};
d.prototype.slide=function(m,i){var p=this.$element.find(".item.active");
var g=i||this.getItemForDirection(m,p);
var k=this.interval;
var n=m=="next"?"left":"right";
var j=this;
if(g.hasClass("active")){return(this.sliding=false)
}var l=g[0];
var f=c.Event("slide.bs.carousel",{relatedTarget:l,direction:n});
this.$element.trigger(f);
if(f.isDefaultPrevented()){return
}this.sliding=true;
k&&this.pause();
if(this.$indicators.length){this.$indicators.find(".active").removeClass("active");
var h=c(this.$indicators.children()[this.getItemIndex(g)]);
h&&h.addClass("active")
}var o=c.Event("slid.bs.carousel",{relatedTarget:l,direction:n});
if(c.support.transition&&this.$element.hasClass("slide")){g.addClass(m);
g[0].offsetWidth;
p.addClass(n);
g.addClass(n);
p.one("bsTransitionEnd",function(){g.removeClass([m,n].join(" ")).addClass("active");
p.removeClass(["active",n].join(" "));
j.sliding=false;
setTimeout(function(){j.$element.trigger(o)
},0)
}).emulateTransitionEnd(d.TRANSITION_DURATION)
}else{p.removeClass("active");
g.addClass("active");
this.sliding=false;
this.$element.trigger(o)
}k&&this.cycle();
return this
};
function b(f){return this.each(function(){var j=c(this);
var i=j.data("bs.carousel");
var g=c.extend({},d.DEFAULTS,j.data(),typeof f=="object"&&f);
var h=typeof f=="string"?f:g.slide;
if(!i){j.data("bs.carousel",(i=new d(this,g)))
}if(typeof f=="number"){i.to(f)
}else{if(h){i[h]()
}else{if(g.interval){i.pause().cycle()
}}}})
}var a=c.fn.carousel;
c.fn.carousel=b;
c.fn.carousel.Constructor=d;
c.fn.carousel.noConflict=function(){c.fn.carousel=a;
return this
};
var e=function(k){var g;
var j=c(this);
var f=c(j.attr("data-target")||(g=j.attr("href"))&&g.replace(/.*(?=#[^\s]+$)/,""));
if(!f.hasClass("carousel")){return
}var h=c.extend({},f.data(),j.data());
var i=j.attr("data-slide-to");
if(i){h.interval=false
}b.call(f,h);
if(i){f.data("bs.carousel").to(i)
}k.preventDefault()
};
c(document).on("click.bs.carousel.data-api","[data-slide]",e).on("click.bs.carousel.data-api","[data-slide-to]",e);
c(window).on("load",function(){c('[data-ride="carousel"]').each(function(){var f=c(this);
b.call(f,f.data())
})
})
}(jQuery);
!function(a){"function"==typeof define&&define.amd?define(["jquery"],a):"undefined"!=typeof module&&module.exports?module.exports=a(require("jquery")):a(jQuery)
}(function(l){var j=-1,c=-1,d=function(a){return parseFloat(a)||0
},k=function(q){var s=1,h=l(q),n=null,p=[];
return h.each(function(){var r=l(this),i=r.offset().top-d(r.css("margin-top")),o=p.length>0?p[p.length-1]:null;
null===o?p.push(r):Math.floor(Math.abs(n-i))<=s?p[p.length-1]=o.add(r):p.push(r),n=i
}),p
},f=function(a){var h={byRow:!0,property:"height",target:null,remove:!1};
return"object"==typeof a?l.extend(h,a):("boolean"==typeof a?h.byRow=a:"remove"===a&&(h.remove=!0),h)
},b=l.fn.matchHeight=function(a){var h=f(a);
if(h.remove){var i=this;
return this.css(h.property,""),l.each(b._groups,function(n,o){o.elements=o.elements.not(i)
}),this
}return this.length<=1&&!h.target?this:(b._groups.push({elements:this,options:h}),b._apply(this,h),this)
};
b.version="0.7.2",b._groups=[],b._throttle=80,b._maintainScroll=!1,b._beforeUpdate=null,b._afterUpdate=null,b._rows=k,b._parse=d,b._parseOptions=f,b._apply=function(t,v){var q=f(v),n=l(t),a=[n],w=l(window).scrollTop(),r=l("html").outerHeight(!0),i=n.parents().filter(":hidden");
return i.each(function(){var h=l(this);
h.data("style-cache",h.attr("style"))
}),i.css("display","block"),q.byRow&&!q.target&&(n.each(function(){var h=l(this),p=h.css("display");
"inline-block"!==p&&"flex"!==p&&"inline-flex"!==p&&(p="block"),h.data("style-cache",h.attr("style")),h.css({display:p,"padding-top":"0","padding-bottom":"0","margin-top":"0","margin-bottom":"0","border-top-width":"0","border-bottom-width":"0",height:"100px",overflow:"hidden"})
}),a=k(n),n.each(function(){var h=l(this);
h.attr("style",h.data("style-cache")||"")
})),l.each(a,function(s,u){var h=l(u),p=0;
if(q.target){p=q.target.outerHeight(!1)
}else{if(q.byRow&&h.length<=1){return void h.css(q.property,"")
}h.each(function(){var y=l(this),z=y.attr("style"),A=y.css("display");
"inline-block"!==A&&"flex"!==A&&"inline-flex"!==A&&(A="block");
var x={display:A};
x[q.property]="",y.css(x),y.outerHeight(!1)>p&&(p=y.outerHeight(!1)),z?y.attr("style",z):y.css("display","")
})
}h.each(function(){var x=l(this),y=0;
q.target&&x.is(q.target)||("border-box"!==x.css("box-sizing")&&(y+=d(x.css("border-top-width"))+d(x.css("border-bottom-width")),y+=d(x.css("padding-top"))+d(x.css("padding-bottom"))),x.css(q.property,p-y+"px"))
})
}),i.each(function(){var h=l(this);
h.attr("style",h.data("style-cache")||null)
}),b._maintainScroll&&l(window).scrollTop(w/r*l("html").outerHeight(!0)),this
},b._applyDataApi=function(){var a={};
l("[data-match-height], [data-mh]").each(function(){var e=l(this),h=e.attr("data-mh")||e.attr("data-match-height");
h in a?a[h]=a[h].add(e):a[h]=e
}),l.each(a,function(){this.matchHeight(!0)
})
};
var m=function(a){b._beforeUpdate&&b._beforeUpdate(a,b._groups),l.each(b._groups,function(){b._apply(this.elements,this.options)
}),b._afterUpdate&&b._afterUpdate(a,b._groups)
};
b._update=function(o,e){if(e&&"resize"===e.type){var h=l(window).width();
if(h===j){return
}j=h
}o?c===-1&&(c=setTimeout(function(){m(e),c=-1
},b._throttle)):m(e)
},l(b._applyDataApi);
var g=l.fn.on?"on":"bind";
l(window)[g]("load",function(a){b._update(!1,a)
}),l(window)[g]("resize orientationchange",function(a){b._update(!0,a)
})
});
!function(g,f,j,i){function h(a,d){this.settings=null,this.options=g.extend({},h.Defaults,d),this.$element=g(a),this._handlers={},this._plugins={},this._supress={},this._current=null,this._speed=null,this._coordinates=[],this._breakpoint=null,this._width=null,this._items=[],this._clones=[],this._mergers=[],this._widths=[],this._invalidated={},this._pipe=[],this._drag={time:null,target:null,pointer:null,stage:{start:null,current:null},direction:null},this._states={current:{},tags:{initializing:["busy"],animating:["busy"],dragging:["interacting"]}},g.each(["onResize","onThrottledResize"],g.proxy(function(e,k){this._handlers[k]=g.proxy(this[k],this)
},this)),g.each(h.Plugins,g.proxy(function(e,c){this._plugins[e.charAt(0).toLowerCase()+e.slice(1)]=new c(this)
},this)),g.each(h.Workers,g.proxy(function(e,k){this._pipe.push({filter:k.filter,run:g.proxy(k.run,this)})
},this)),this.setup(),this.initialize()
}h.Defaults={items:3,loop:!1,center:!1,rewind:!1,mouseDrag:!0,touchDrag:!0,pullDrag:!0,freeDrag:!1,margin:0,stagePadding:0,merge:!1,mergeFit:!0,autoWidth:!1,startPosition:0,rtl:!1,smartSpeed:250,fluidSpeed:!1,dragEndSpeed:!1,responsive:{},responsiveRefreshRate:200,responsiveBaseElement:f,fallbackEasing:"swing",info:!1,nestedItemSelector:!1,itemElement:"div",stageElement:"div",refreshClass:"owl-refresh",loadedClass:"owl-loaded",loadingClass:"owl-loading",rtlClass:"owl-rtl",responsiveClass:"owl-responsive",dragClass:"owl-drag",itemClass:"owl-item",stageClass:"owl-stage",stageOuterClass:"owl-stage-outer",grabClass:"owl-grab"},h.Width={Default:"default",Inner:"inner",Outer:"outer"},h.Type={Event:"event",State:"state"},h.Plugins={},h.Workers=[{filter:["width","settings"],run:function(){this._width=this.$element.width()
}},{filter:["width","items","settings"],run:function(b){b.current=this._items&&this._items[this.relative(this._current)]
}},{filter:["items","settings"],run:function(){this.$stage.children(".cloned").remove()
}},{filter:["width","items","settings"],run:function(l){var k=this.settings.margin||"",o=!this.settings.autoWidth,n=this.settings.rtl,m={width:"auto","margin-left":n?k:"","margin-right":n?"":k};
!o&&this.$stage.children().css(m),l.css=m
}},{filter:["width","items","settings"],run:function(l){var k=(this.width()/this.settings.items).toFixed(3)-this.settings.margin,p=null,o=this._items.length,n=!this.settings.autoWidth,m=[];
for(l.items={merge:!1,width:k};
o--;
){p=this._mergers[o],p=this.settings.mergeFit&&Math.min(p,this.settings.items)||p,l.items.merge=p>1||l.items.merge,m[o]=n?k*p:this._items[o].width()
}this._widths=m
}},{filter:["items","settings"],run:function(){var a=[],q=this._items,p=this.settings,o=Math.max(2*p.items,4),n=2*Math.ceil(q.length/2),m=p.loop&&q.length?p.rewind?o:Math.max(o,n):0,l="",k="";
for(m/=2;
m--;
){a.push(this.normalize(a.length/2,!0)),l+=q[a[a.length-1]][0].outerHTML,a.push(this.normalize(q.length-1-(a.length-1)/2,!0)),k=q[a[a.length-1]][0].outerHTML+k
}this._clones=a,g(l).addClass("cloned").appendTo(this.$stage),g(k).addClass("cloned").prependTo(this.$stage)
}},{filter:["width","items","settings"],run:function(){for(var l=this.settings.rtl?1:-1,k=this._clones.length+this._items.length,p=-1,o=0,n=0,m=[];
++p<k;
){o=m[p-1]||0,n=this._widths[this.relative(p)]+this.settings.margin,m.push(o+n*l)
}this._coordinates=m
}},{filter:["width","items","settings"],run:function(){var e=this.settings.stagePadding,d=this._coordinates,k={width:Math.ceil(Math.abs(d[d.length-1]))+2*e,"padding-left":e||"","padding-right":e||""};
this.$stage.css(k)
}},{filter:["width","items","settings"],run:function(k){var e=this._coordinates.length,m=!this.settings.autoWidth,l=this.$stage.children();
if(m&&k.items.merge){for(;
e--;
){k.css.width=this._widths[this.relative(e)],l.eq(e).css(k.css)
}}else{m&&(k.css.width=k.items.width,l.css(k.css))
}}},{filter:["items"],run:function(){this._coordinates.length<1&&this.$stage.removeAttr("style")
}},{filter:["width","items","settings"],run:function(b){b.current=b.current?this.$stage.children().index(b.current):0,b.current=Math.max(this.minimum(),Math.min(this.maximum(),b.current)),this.reset(b.current)
}},{filter:["position"],run:function(){this.animate(this.coordinates(this._current))
}},{filter:["width","position","items","settings"],run:function(){var s,r,q,p,o=this.settings.rtl?1:-1,n=2*this.settings.stagePadding,m=this.coordinates(this.current())+n,l=m+this.width()*o,k=[];
for(q=0,p=this._coordinates.length;
q<p;
q++){s=this._coordinates[q-1]||0,r=Math.abs(this._coordinates[q])+n*o,(this.op(s,"<=",m)&&this.op(s,">",l)||this.op(r,"<",m)&&this.op(r,">",l))&&k.push(q)
}this.$stage.children(".active").removeClass("active"),this.$stage.children(":eq("+k.join("), :eq(")+")").addClass("active"),this.settings.center&&(this.$stage.children(".center").removeClass("center"),this.$stage.children().eq(this.current()).addClass("center"))
}}],h.prototype.initialize=function(){if(this.enter("initializing"),this.trigger("initialize"),this.$element.toggleClass(this.settings.rtlClass,this.settings.rtl),this.settings.autoWidth&&!this.is("pre-loading")){var a,k,d;
a=this.$element.find("img"),k=this.settings.nestedItemSelector?"."+this.settings.nestedItemSelector:i,d=this.$element.children(k).width(),a.length&&d<=0&&this.preloadAutoWidthImages(a)
}this.$element.addClass(this.options.loadingClass),this.$stage=g("<"+this.settings.stageElement+' class="'+this.settings.stageClass+'"/>').wrap('<div class="'+this.settings.stageOuterClass+'"/>'),this.$element.append(this.$stage.parent()),this.replace(this.$element.children().not(this.$stage.parent())),this.$element.is(":visible")?this.refresh():this.invalidate("width"),this.$element.removeClass(this.options.loadingClass).addClass(this.options.loadedClass),this.registerEventHandlers(),this.leave("initializing"),this.trigger("initialized")
},h.prototype.setup=function(){var a=this.viewport(),m=this.options.responsive,l=-1,k=null;
m?(g.each(m,function(b){b<=a&&b>l&&(l=Number(b))
}),k=g.extend({},this.options,m[l]),"function"==typeof k.stagePadding&&(k.stagePadding=k.stagePadding()),delete k.responsive,k.responsiveClass&&this.$element.attr("class",this.$element.attr("class").replace(new RegExp("("+this.options.responsiveClass+"-)\\S+\\s","g"),"$1"+l))):k=g.extend({},this.options),this.trigger("change",{property:{name:"settings",value:k}}),this._breakpoint=l,this.settings=k,this.invalidate("settings"),this.trigger("changed",{property:{name:"settings",value:this.settings}})
},h.prototype.optionsLogic=function(){this.settings.autoWidth&&(this.settings.stagePadding=!1,this.settings.merge=!1)
},h.prototype.prepare=function(a){var d=this.trigger("prepare",{content:a});
return d.data||(d.data=g("<"+this.settings.itemElement+"/>").addClass(this.options.itemClass).append(a)),this.trigger("prepared",{content:d.data}),d.data
},h.prototype.update=function(){for(var a=0,m=this._pipe.length,l=g.proxy(function(b){return this[b]
},this._invalidated),k={};
a<m;
){(this._invalidated.all||g.grep(this._pipe[a].filter,l).length>0)&&this._pipe[a].run(k),a++
}this._invalidated={},!this.is("valid")&&this.enter("valid")
},h.prototype.width=function(b){switch(b=b||h.Width.Default){case h.Width.Inner:case h.Width.Outer:return this._width;
default:return this._width-2*this.settings.stagePadding+this.settings.margin
}},h.prototype.refresh=function(){this.enter("refreshing"),this.trigger("refresh"),this.setup(),this.optionsLogic(),this.$element.addClass(this.options.refreshClass),this.update(),this.$element.removeClass(this.options.refreshClass),this.leave("refreshing"),this.trigger("refreshed")
},h.prototype.onThrottledResize=function(){f.clearTimeout(this.resizeTimer),this.resizeTimer=f.setTimeout(this._handlers.onResize,this.settings.responsiveRefreshRate)
},h.prototype.onResize=function(){return !!this._items.length&&(this._width!==this.$element.width()&&(!!this.$element.is(":visible")&&(this.enter("resizing"),this.trigger("resize").isDefaultPrevented()?(this.leave("resizing"),!1):(this.invalidate("width"),this.refresh(),this.leave("resizing"),void this.trigger("resized")))))
},h.prototype.registerEventHandlers=function(){g.support.transition&&this.$stage.on(g.support.transition.end+".owl.core",g.proxy(this.onTransitionEnd,this)),this.settings.responsive!==!1&&this.on(f,"resize",this._handlers.onThrottledResize),this.settings.mouseDrag&&(this.$element.addClass(this.options.dragClass),this.$stage.on("mousedown.owl.core",g.proxy(this.onDragStart,this)),this.$stage.on("dragstart.owl.core selectstart.owl.core",function(){return !1
})),this.settings.touchDrag&&(this.$stage.on("touchstart.owl.core",g.proxy(this.onDragStart,this)),this.$stage.on("touchcancel.owl.core",g.proxy(this.onDragEnd,this)))
},h.prototype.onDragStart=function(a){var c=null;
3!==a.which&&(g.support.transform?(c=this.$stage.css("transform").replace(/.*\(|\)| /g,"").split(","),c={x:c[16===c.length?12:4],y:c[16===c.length?13:5]}):(c=this.$stage.position(),c={x:this.settings.rtl?c.left+this.$stage.width()-this.width()+this.settings.margin:c.left,y:c.top}),this.is("animating")&&(g.support.transform?this.animate(c.x):this.$stage.stop(),this.invalidate("position")),this.$element.toggleClass(this.options.grabClass,"mousedown"===a.type),this.speed(0),this._drag.time=(new Date).getTime(),this._drag.target=g(a.target),this._drag.stage.start=c,this._drag.stage.current=c,this._drag.pointer=this.pointer(a),g(j).on("mouseup.owl.core touchend.owl.core",g.proxy(this.onDragEnd,this)),g(j).one("mousemove.owl.core touchmove.owl.core",g.proxy(function(e){var k=this.difference(this._drag.pointer,this.pointer(e));
g(j).on("mousemove.owl.core touchmove.owl.core",g.proxy(this.onDragMove,this)),Math.abs(k.x)<Math.abs(k.y)&&this.is("valid")||(e.preventDefault(),this.enter("dragging"),this.trigger("drag"))
},this)))
},h.prototype.onDragMove=function(l){var k=null,p=null,o=null,n=this.difference(this._drag.pointer,this.pointer(l)),m=this.difference(this._drag.stage.start,n);
this.is("dragging")&&(l.preventDefault(),this.settings.loop?(k=this.coordinates(this.minimum()),p=this.coordinates(this.maximum()+1)-k,m.x=((m.x-k)%p+p)%p+k):(k=this.settings.rtl?this.coordinates(this.maximum()):this.coordinates(this.minimum()),p=this.settings.rtl?this.coordinates(this.minimum()):this.coordinates(this.maximum()),o=this.settings.pullDrag?-1*n.x/5:0,m.x=Math.max(Math.min(m.x,k+o),p+o)),this._drag.stage.current=m,this.animate(m.x))
},h.prototype.onDragEnd=function(a){var l=this.difference(this._drag.pointer,this.pointer(a)),k=this._drag.stage.current,c=l.x>0^this.settings.rtl?"left":"right";
g(j).off(".owl.core"),this.$element.removeClass(this.options.grabClass),(0!==l.x&&this.is("dragging")||!this.is("valid"))&&(this.speed(this.settings.dragEndSpeed||this.settings.smartSpeed),this.current(this.closest(k.x,0!==l.x?c:this._drag.direction)),this.invalidate("position"),this.update(),this._drag.direction=c,(Math.abs(l.x)>3||(new Date).getTime()-this._drag.time>300)&&this._drag.target.one("click.owl.core",function(){return !1
})),this.is("dragging")&&(this.leave("dragging"),this.trigger("dragged"))
},h.prototype.closest=function(a,o){var n=-1,m=30,l=this.width(),k=this.coordinates();
return this.settings.freeDrag||g.each(k,g.proxy(function(b,c){return"left"===o&&a>c-m&&a<c+m?n=b:"right"===o&&a>c-l-m&&a<c-l+m?n=b+1:this.op(a,"<",c)&&this.op(a,">",k[b+1]||c-l)&&(n="left"===o?b+1:b),n===-1
},this)),this.settings.loop||(this.op(a,">",k[this.minimum()])?n=a=this.minimum():this.op(a,"<",k[this.maximum()])&&(n=a=this.maximum())),n
},h.prototype.animate=function(a){var d=this.speed()>0;
this.is("animating")&&this.onTransitionEnd(),d&&(this.enter("animating"),this.trigger("translate")),g.support.transform3d&&g.support.transition?this.$stage.css({transform:"translate3d("+a+"px,0px,0px)",transition:this.speed()/1000+"s"}):d?this.$stage.animate({left:a+"px"},this.speed(),this.settings.fallbackEasing,g.proxy(this.onTransitionEnd,this)):this.$stage.css({left:a+"px"})
},h.prototype.is=function(b){return this._states.current[b]&&this._states.current[b]>0
},h.prototype.current=function(d){if(d===i){return this._current
}if(0===this._items.length){return i
}if(d=this.normalize(d),this._current!==d){var c=this.trigger("change",{property:{name:"position",value:d}});
c.data!==i&&(d=this.normalize(c.data)),this._current=d,this.invalidate("position"),this.trigger("changed",{property:{name:"position",value:this._current}})
}return this._current
},h.prototype.invalidate=function(a){return"string"===g.type(a)&&(this._invalidated[a]=!0,this.is("valid")&&this.leave("valid")),g.map(this._invalidated,function(d,c){return c
})
},h.prototype.reset=function(b){b=this.normalize(b),b!==i&&(this._speed=0,this._current=b,this.suppress(["translate","translated"]),this.animate(this.coordinates(b)),this.release(["translate","translated"]))
},h.prototype.normalize=function(k,d){var m=this._items.length,l=d?0:this._clones.length;
return !this.isNumeric(k)||m<1?k=i:(k<0||k>=m+l)&&(k=((k-l/2)%m+m)%m+l/2),k
},h.prototype.relative=function(b){return b-=this._clones.length/2,this.normalize(b,!0)
},h.prototype.maximum=function(l){var k,p,o,n=this.settings,m=this._coordinates.length;
if(n.loop){m=this._clones.length/2+this._items.length-1
}else{if(n.autoWidth||n.merge){for(k=this._items.length,p=this._items[--k].width(),o=this.$element.width();
k--&&(p+=this._items[k].width()+this.settings.margin,!(p>o));
){}m=k+1
}else{m=n.center?this._items.length-1:this._items.length-n.items
}}return l&&(m-=this._clones.length/2),Math.max(m,0)
},h.prototype.minimum=function(b){return b?0:this._clones.length/2
},h.prototype.items=function(b){return b===i?this._items.slice():(b=this.normalize(b,!0),this._items[b])
},h.prototype.mergers=function(b){return b===i?this._mergers.slice():(b=this.normalize(b,!0),this._mergers[b])
},h.prototype.clones=function(a){var l=this._clones.length/2,k=l+this._items.length,d=function(b){return b%2===0?k+b/2:l-(b+1)/2
};
return a===i?g.map(this._clones,function(e,c){return d(c)
}):g.map(this._clones,function(b,e){return b===a?d(e):null
})
},h.prototype.speed=function(b){return b!==i&&(this._speed=b),this._speed
},h.prototype.coordinates=function(a){var l,k=1,d=a-1;
return a===i?g.map(this._coordinates,g.proxy(function(e,c){return this.coordinates(c)
},this)):(this.settings.center?(this.settings.rtl&&(k=-1,d=a+1),l=this._coordinates[a],l+=(this.width()-l+(this._coordinates[d]||0))/2*k):l=this._coordinates[d]||0,l=Math.ceil(l))
},h.prototype.duration=function(e,d,k){return 0===k?0:Math.min(Math.max(Math.abs(d-e),1),6)*Math.abs(k||this.settings.smartSpeed)
},h.prototype.to=function(s,r){var q=this.current(),p=null,o=s-this.relative(q),n=(o>0)-(o<0),m=this._items.length,l=this.minimum(),k=this.maximum();
this.settings.loop?(!this.settings.rewind&&Math.abs(o)>m/2&&(o+=n*-1*m),s=q+o,p=((s-l)%m+m)%m+l,p!==s&&p-o<=k&&p-o>0&&(q=p-o,s=p,this.reset(q))):this.settings.rewind?(k+=1,s=(s%k+k)%k):s=Math.max(l,Math.min(k,s)),this.speed(this.duration(q,s,r)),this.current(s),this.$element.is(":visible")&&this.update()
},h.prototype.next=function(b){b=b||!1,this.to(this.relative(this.current())+1,b)
},h.prototype.prev=function(b){b=b||!1,this.to(this.relative(this.current())-1,b)
},h.prototype.onTransitionEnd=function(b){if(b!==i&&(b.stopPropagation(),(b.target||b.srcElement||b.originalTarget)!==this.$stage.get(0))){return !1
}this.leave("animating"),this.trigger("translated")
},h.prototype.viewport=function(){var a;
return this.options.responsiveBaseElement!==f?a=g(this.options.responsiveBaseElement).width():f.innerWidth?a=f.innerWidth:j.documentElement&&j.documentElement.clientWidth?a=j.documentElement.clientWidth:console.warn("Can not detect viewport width."),a
},h.prototype.replace=function(a){this.$stage.empty(),this._items=[],a&&(a=a instanceof jQuery?a:g(a)),this.settings.nestedItemSelector&&(a=a.find("."+this.settings.nestedItemSelector)),a.filter(function(){return 1===this.nodeType
}).each(g.proxy(function(d,c){c=this.prepare(c),this.$stage.append(c),this._items.push(c),this._mergers.push(1*c.find("[data-merge]").addBack("[data-merge]").attr("data-merge")||1)
},this)),this.reset(this.isNumeric(this.settings.startPosition)?this.settings.startPosition:0),this.invalidate("items")
},h.prototype.add=function(a,k){var d=this.relative(this._current);
k=k===i?this._items.length:this.normalize(k,!0),a=a instanceof jQuery?a:g(a),this.trigger("add",{content:a,position:k}),a=this.prepare(a),0===this._items.length||k===this._items.length?(0===this._items.length&&this.$stage.append(a),0!==this._items.length&&this._items[k-1].after(a),this._items.push(a),this._mergers.push(1*a.find("[data-merge]").addBack("[data-merge]").attr("data-merge")||1)):(this._items[k].before(a),this._items.splice(k,0,a),this._mergers.splice(k,0,1*a.find("[data-merge]").addBack("[data-merge]").attr("data-merge")||1)),this._items[d]&&this.reset(this._items[d].index()),this.invalidate("items"),this.trigger("added",{content:a,position:k})
},h.prototype.remove=function(b){b=this.normalize(b,!0),b!==i&&(this.trigger("remove",{content:this._items[b],position:b}),this._items[b].remove(),this._items.splice(b,1),this._mergers.splice(b,1),this.invalidate("items"),this.trigger("removed",{content:null,position:b}))
},h.prototype.preloadAutoWidthImages=function(a){a.each(g.proxy(function(d,e){this.enter("pre-loading"),e=g(e),g(new Image).one("load",g.proxy(function(b){e.attr("src",b.target.src),e.css("opacity",1),this.leave("pre-loading"),!this.is("pre-loading")&&!this.is("initializing")&&this.refresh()
},this)).attr("src",e.attr("src")||e.attr("data-src")||e.attr("data-src-retina"))
},this))
},h.prototype.destroy=function(){this.$element.off(".owl.core"),this.$stage.off(".owl.core"),g(j).off(".owl.core"),this.settings.responsive!==!1&&(f.clearTimeout(this.resizeTimer),this.off(f,"resize",this._handlers.onThrottledResize));
for(var a in this._plugins){this._plugins[a].destroy()
}this.$stage.children(".cloned").remove(),this.$stage.unwrap(),this.$stage.children().contents().unwrap(),this.$stage.children().unwrap(),this.$element.removeClass(this.options.refreshClass).removeClass(this.options.loadingClass).removeClass(this.options.loadedClass).removeClass(this.options.rtlClass).removeClass(this.options.dragClass).removeClass(this.options.grabClass).attr("class",this.$element.attr("class").replace(new RegExp(this.options.responsiveClass+"-\\S+\\s","g"),"")).removeData("owl.carousel")
},h.prototype.op=function(k,e,m){var l=this.settings.rtl;
switch(e){case"<":return l?k>m:k<m;
case">":return l?k<m:k>m;
case">=":return l?k<=m:k>=m;
case"<=":return l?k>=m:k<=m
}},h.prototype.on=function(k,e,m,l){k.addEventListener?k.addEventListener(e,m,l):k.attachEvent&&k.attachEvent("on"+e,m)
},h.prototype.off=function(k,e,m,l){k.removeEventListener?k.removeEventListener(e,m,l):k.detachEvent&&k.detachEvent("on"+e,m)
},h.prototype.trigger=function(a,p,o,n,m){var l={item:{count:this._items.length,index:this.current()}},k=g.camelCase(g.grep(["on",a,o],function(b){return b
}).join("-").toLowerCase()),e=g.Event([a,"owl",o||"carousel"].join(".").toLowerCase(),g.extend({relatedTarget:this},l,p));
return this._supress[a]||(g.each(this._plugins,function(d,c){c.onTrigger&&c.onTrigger(e)
}),this.register({type:h.Type.Event,name:a}),this.$element.trigger(e),this.settings&&"function"==typeof this.settings[k]&&this.settings[k].call(this,e)),e
},h.prototype.enter=function(a){g.each([a].concat(this._states.tags[a]||[]),g.proxy(function(d,c){this._states.current[c]===i&&(this._states.current[c]=0),this._states.current[c]++
},this))
},h.prototype.leave=function(a){g.each([a].concat(this._states.tags[a]||[]),g.proxy(function(d,c){this._states.current[c]--
},this))
},h.prototype.register=function(a){if(a.type===h.Type.Event){if(g.event.special[a.name]||(g.event.special[a.name]={}),!g.event.special[a.name].owl){var d=g.event.special[a.name]._default;
g.event.special[a.name]._default=function(b){return !d||!d.apply||b.namespace&&b.namespace.indexOf("owl")!==-1?b.namespace&&b.namespace.indexOf("owl")>-1:d.apply(this,arguments)
},g.event.special[a.name].owl=!0
}}else{a.type===h.Type.State&&(this._states.tags[a.name]?this._states.tags[a.name]=this._states.tags[a.name].concat(a.tags):this._states.tags[a.name]=a.tags,this._states.tags[a.name]=g.grep(this._states.tags[a.name],g.proxy(function(e,b){return g.inArray(e,this._states.tags[a.name])===b
},this)))
}},h.prototype.suppress=function(a){g.each(a,g.proxy(function(d,c){this._supress[c]=!0
},this))
},h.prototype.release=function(a){g.each(a,g.proxy(function(d,c){delete this._supress[c]
},this))
},h.prototype.pointer=function(b){var d={x:null,y:null};
return b=b.originalEvent||b||f.event,b=b.touches&&b.touches.length?b.touches[0]:b.changedTouches&&b.changedTouches.length?b.changedTouches[0]:b,b.pageX?(d.x=b.pageX,d.y=b.pageY):(d.x=b.clientX,d.y=b.clientY),d
},h.prototype.isNumeric=function(b){return !isNaN(parseFloat(b))
},h.prototype.difference=function(d,c){return{x:d.x-c.x,y:d.y-c.y}
},g.fn.owlCarousel=function(a){var d=Array.prototype.slice.call(arguments,1);
return this.each(function(){var c=g(this),b=c.data("owl.carousel");
b||(b=new h(this,"object"==typeof a&&a),c.data("owl.carousel",b),g.each(["next","prev","to","destroy","refresh","replace","add","remove"],function(e,k){b.register({type:h.Type.Event,name:k}),b.$element.on(k+".owl.carousel.core",g.proxy(function(l){l.namespace&&l.relatedTarget!==this&&(this.suppress([k]),b[k].apply(this,[].slice.call(arguments,1)),this.release([k]))
},b))
})),"string"==typeof a&&"_"!==a.charAt(0)&&b[a].apply(b,d)
})
},g.fn.owlCarousel.Constructor=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this._core=a,this._interval=null,this._visible=null,this._handlers={"initialized.owl.carousel":g.proxy(function(b){b.namespace&&this._core.settings.autoRefresh&&this.watch()
},this)},this._core.options=g.extend({},h.Defaults,this._core.options),this._core.$element.on(this._handlers)
};
h.Defaults={autoRefresh:!0,autoRefreshInterval:500},h.prototype.watch=function(){this._interval||(this._visible=this._core.$element.is(":visible"),this._interval=f.setInterval(g.proxy(this.refresh,this),this._core.settings.autoRefreshInterval))
},h.prototype.refresh=function(){this._core.$element.is(":visible")!==this._visible&&(this._visible=!this._visible,this._core.$element.toggleClass("owl-hidden",!this._visible),this._visible&&this._core.invalidate("width")&&this._core.refresh())
},h.prototype.destroy=function(){var b,d;
f.clearInterval(this._interval);
for(b in this._handlers){this._core.$element.off(b,this._handlers[b])
}for(d in Object.getOwnPropertyNames(this)){"function"!=typeof this[d]&&(this[d]=null)
}},g.fn.owlCarousel.Constructor.Plugins.AutoRefresh=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this._core=a,this._loaded=[],this._handlers={"initialized.owl.carousel change.owl.carousel resized.owl.carousel":g.proxy(function(d){if(d.namespace&&this._core.settings&&this._core.settings.lazyLoad&&(d.property&&"position"==d.property.name||"initialized"==d.type)){for(var p=this._core.settings,o=p.center&&Math.ceil(p.items/2)||p.items,n=p.center&&o*-1||0,m=(d.property&&d.property.value!==i?d.property.value:this._core.current())+n,l=this._core.clones().length,k=g.proxy(function(e,c){this.load(c)
},this);
n++<o;
){this.load(l/2+this._core.relative(m)),l&&g.each(this._core.clones(this._core.relative(m)),k),m++
}}},this)},this._core.options=g.extend({},h.Defaults,this._core.options),this._core.$element.on(this._handlers)
};
h.Defaults={lazyLoad:!1},h.prototype.load=function(k){var b=this._core.$stage.children().eq(k),a=b&&b.find(".owl-lazy");
!a||g.inArray(b.get(0),this._loaded)>-1||(a.each(g.proxy(function(p,o){var n,m=g(o),l=f.devicePixelRatio>1&&m.attr("data-src-retina")||m.attr("data-src");
this._core.trigger("load",{element:m,url:l},"lazy"),m.is("img")?m.one("load.owl.lazy",g.proxy(function(){m.css("opacity",1),this._core.trigger("loaded",{element:m,url:l},"lazy")
},this)).attr("src",l):(n=new Image,n.onload=g.proxy(function(){m.css({"background-image":'url("'+l+'")',opacity:"1"}),this._core.trigger("loaded",{element:m,url:l},"lazy")
},this),n.src=l)
},this)),this._loaded.push(b.get(0)))
},h.prototype.destroy=function(){var d,c;
for(d in this.handlers){this._core.$element.off(d,this.handlers[d])
}for(c in Object.getOwnPropertyNames(this)){"function"!=typeof this[c]&&(this[c]=null)
}},g.fn.owlCarousel.Constructor.Plugins.Lazy=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this._core=a,this._handlers={"initialized.owl.carousel refreshed.owl.carousel":g.proxy(function(b){b.namespace&&this._core.settings.autoHeight&&this.update()
},this),"changed.owl.carousel":g.proxy(function(b){b.namespace&&this._core.settings.autoHeight&&"position"==b.property.name&&this.update()
},this),"loaded.owl.lazy":g.proxy(function(b){b.namespace&&this._core.settings.autoHeight&&b.element.closest("."+this._core.settings.itemClass).index()===this._core.current()&&this.update()
},this)},this._core.options=g.extend({},h.Defaults,this._core.options),this._core.$element.on(this._handlers)
};
h.Defaults={autoHeight:!1,autoHeightClass:"owl-height"},h.prototype.update=function(){var a=this._core._current,n=a+this._core.settings.items,m=this._core.$stage.children().toArray().slice(a,n),l=[],k=0;
g.each(m,function(d,e){l.push(g(e).height())
}),k=Math.max.apply(null,l),this._core.$stage.parent().height(k).addClass(this._core.settings.autoHeightClass)
},h.prototype.destroy=function(){var d,c;
for(d in this._handlers){this._core.$element.off(d,this._handlers[d])
}for(c in Object.getOwnPropertyNames(this)){"function"!=typeof this[c]&&(this[c]=null)
}},g.fn.owlCarousel.Constructor.Plugins.AutoHeight=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this._core=a,this._videos={},this._playing=null,this._handlers={"initialized.owl.carousel":g.proxy(function(b){b.namespace&&this._core.register({type:"state",name:"playing",tags:["interacting"]})
},this),"resize.owl.carousel":g.proxy(function(b){b.namespace&&this._core.settings.video&&this.isInFullScreen()&&b.preventDefault()
},this),"refreshed.owl.carousel":g.proxy(function(b){b.namespace&&this._core.is("resizing")&&this._core.$stage.find(".cloned .owl-video-frame").remove()
},this),"changed.owl.carousel":g.proxy(function(b){b.namespace&&"position"===b.property.name&&this._playing&&this.stop()
},this),"prepared.owl.carousel":g.proxy(function(d){if(d.namespace){var e=g(d.content).find(".owl-video");
e.length&&(e.css("display","none"),this.fetch(e,g(d.content)))
}},this)},this._core.options=g.extend({},h.Defaults,this._core.options),this._core.$element.on(this._handlers),this._core.$element.on("click.owl.video",".owl-video-play-icon",g.proxy(function(b){this.play(b)
},this))
};
h.Defaults={video:!1,videoHeight:!1,videoWidth:!1},h.prototype.fetch=function(l,k){var q=function(){return l.attr("data-vimeo-id")?"vimeo":l.attr("data-vzaar-id")?"vzaar":"youtube"
}(),p=l.attr("data-vimeo-id")||l.attr("data-youtube-id")||l.attr("data-vzaar-id"),o=l.attr("data-width")||this._core.settings.videoWidth,n=l.attr("data-height")||this._core.settings.videoHeight,m=l.attr("href");
if(!m){throw new Error("Missing video URL.")
}if(p=m.match(/(http:|https:|)\/\/(player.|www.|app.)?(vimeo\.com|youtu(be\.com|\.be|be\.googleapis\.com)|vzaar\.com)\/(video\/|videos\/|embed\/|channels\/.+\/|groups\/.+\/|watch\?v=|v\/)?([A-Za-z0-9._%-]*)(\&\S+)?/),p[3].indexOf("youtu")>-1){q="youtube"
}else{if(p[3].indexOf("vimeo")>-1){q="vimeo"
}else{if(!(p[3].indexOf("vzaar")>-1)){throw new Error("Video URL not supported.")
}q="vzaar"
}}p=p[6],this._videos[m]={type:q,id:p,width:o,height:n},k.attr("data-video",m),this.thumbnail(l,this._videos[m])
},h.prototype.thumbnail=function(v,u){var t,s,r,q=u.width&&u.height?'style="width:'+u.width+"px;height:"+u.height+'px;"':"",p=v.find("img"),o="src",n="",m=this._core.settings,a=function(b){s='<div class="owl-video-play-icon"></div>',t=m.lazyLoad?'<div class="owl-video-tn '+n+'" '+o+'="'+b+'"></div>':'<div class="owl-video-tn" style="opacity:1;background-image:url('+b+')"></div>',v.after(t),v.after(s)
};
if(v.wrap('<div class="owl-video-wrapper"'+q+"></div>"),this._core.settings.lazyLoad&&(o="data-src",n="owl-lazy"),p.length){return a(p.attr(o)),p.remove(),!1
}"youtube"===u.type?(r="//img.youtube.com/vi/"+u.id+"/hqdefault.jpg",a(r)):"vimeo"===u.type?g.ajax({type:"GET",url:"//vimeo.com/api/v2/video/"+u.id+".json",jsonp:"callback",dataType:"jsonp",success:function(b){r=b[0].thumbnail_large,a(r)
}}):"vzaar"===u.type&&g.ajax({type:"GET",url:"//vzaar.com/api/videos/"+u.id+".json",jsonp:"callback",dataType:"jsonp",success:function(b){r=b.framegrab_url,a(r)
}})
},h.prototype.stop=function(){this._core.trigger("stop",null,"video"),this._playing.find(".owl-video-frame").remove(),this._playing.removeClass("owl-video-playing"),this._playing=null,this._core.leave("playing"),this._core.trigger("stopped",null,"video")
},h.prototype.play=function(a){var p,o=g(a.target),n=o.closest("."+this._core.settings.itemClass),m=this._videos[n.attr("data-video")],l=m.width||"100%",k=m.height||this._core.$stage.height();
this._playing||(this._core.enter("playing"),this._core.trigger("play",null,"video"),n=this._core.items(this._core.relative(n.index())),this._core.reset(n.index()),"youtube"===m.type?p='<iframe width="'+l+'" height="'+k+'" src="//www.youtube.com/embed/'+m.id+"?autoplay=1&rel=0&v="+m.id+'" frameborder="0" allowfullscreen></iframe>':"vimeo"===m.type?p='<iframe src="//player.vimeo.com/video/'+m.id+'?autoplay=1" width="'+l+'" height="'+k+'" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>':"vzaar"===m.type&&(p='<iframe frameborder="0"height="'+k+'"width="'+l+'" allowfullscreen mozallowfullscreen webkitAllowFullScreen src="//view.vzaar.com/'+m.id+'/player?autoplay=true"></iframe>'),g('<div class="owl-video-frame">'+p+"</div>").insertAfter(n.find(".owl-video")),this._playing=n.addClass("owl-video-playing"))
},h.prototype.isInFullScreen=function(){var a=j.fullscreenElement||j.mozFullScreenElement||j.webkitFullscreenElement;
return a&&g(a).parent().hasClass("owl-video-frame")
},h.prototype.destroy=function(){var d,c;
this._core.$element.off("click.owl.video");
for(d in this._handlers){this._core.$element.off(d,this._handlers[d])
}for(c in Object.getOwnPropertyNames(this)){"function"!=typeof this[c]&&(this[c]=null)
}},g.fn.owlCarousel.Constructor.Plugins.Video=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this.core=a,this.core.options=g.extend({},h.Defaults,this.core.options),this.swapping=!0,this.previous=i,this.next=i,this.handlers={"change.owl.carousel":g.proxy(function(b){b.namespace&&"position"==b.property.name&&(this.previous=this.core.current(),this.next=b.property.value)
},this),"drag.owl.carousel dragged.owl.carousel translated.owl.carousel":g.proxy(function(b){b.namespace&&(this.swapping="translated"==b.type)
},this),"translate.owl.carousel":g.proxy(function(b){b.namespace&&this.swapping&&(this.core.options.animateOut||this.core.options.animateIn)&&this.swap()
},this)},this.core.$element.on(this.handlers)
};
h.Defaults={animateOut:!1,animateIn:!1},h.prototype.swap=function(){if(1===this.core.settings.items&&g.support.animation&&g.support.transition){this.core.speed(0);
var a,o=g.proxy(this.clear,this),n=this.core.$stage.children().eq(this.previous),m=this.core.$stage.children().eq(this.next),l=this.core.settings.animateIn,k=this.core.settings.animateOut;
this.core.current()!==this.previous&&(k&&(a=this.core.coordinates(this.previous)-this.core.coordinates(this.next),n.one(g.support.animation.end,o).css({left:a+"px"}).addClass("animated owl-animated-out").addClass(k)),l&&m.one(g.support.animation.end,o).addClass("animated owl-animated-in").addClass(l))
}},h.prototype.clear=function(a){g(a.target).css({left:""}).removeClass("animated owl-animated-out owl-animated-in").removeClass(this.core.settings.animateIn).removeClass(this.core.settings.animateOut),this.core.onTransitionEnd()
},h.prototype.destroy=function(){var d,c;
for(d in this.handlers){this.core.$element.off(d,this.handlers[d])
}for(c in Object.getOwnPropertyNames(this)){"function"!=typeof this[c]&&(this[c]=null)
}},g.fn.owlCarousel.Constructor.Plugins.Animate=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this._core=a,this._timeout=null,this._paused=!1,this._handlers={"changed.owl.carousel":g.proxy(function(b){b.namespace&&"settings"===b.property.name?this._core.settings.autoplay?this.play():this.stop():b.namespace&&"position"===b.property.name&&this._core.settings.autoplay&&this._setAutoPlayInterval()
},this),"initialized.owl.carousel":g.proxy(function(b){b.namespace&&this._core.settings.autoplay&&this.play()
},this),"play.owl.autoplay":g.proxy(function(e,d,k){e.namespace&&this.play(d,k)
},this),"stop.owl.autoplay":g.proxy(function(b){b.namespace&&this.stop()
},this),"mouseover.owl.autoplay":g.proxy(function(){this._core.settings.autoplayHoverPause&&this._core.is("rotating")&&this.pause()
},this),"mouseleave.owl.autoplay":g.proxy(function(){this._core.settings.autoplayHoverPause&&this._core.is("rotating")&&this.play()
},this),"touchstart.owl.core":g.proxy(function(){this._core.settings.autoplayHoverPause&&this._core.is("rotating")&&this.pause()
},this),"touchend.owl.core":g.proxy(function(){this._core.settings.autoplayHoverPause&&this.play()
},this)},this._core.$element.on(this._handlers),this._core.options=g.extend({},h.Defaults,this._core.options)
};
h.Defaults={autoplay:!1,autoplayTimeout:5000,autoplayHoverPause:!1,autoplaySpeed:!1},h.prototype.play=function(d,c){this._paused=!1,this._core.is("rotating")||(this._core.enter("rotating"),this._setAutoPlayInterval())
},h.prototype._getNextTimeout=function(b,a){return this._timeout&&f.clearTimeout(this._timeout),f.setTimeout(g.proxy(function(){this._paused||this._core.is("busy")||this._core.is("interacting")||j.hidden||this._core.next(a||this._core.settings.autoplaySpeed)
},this),b||this._core.settings.autoplayTimeout)
},h.prototype._setAutoPlayInterval=function(){this._timeout=this._getNextTimeout()
},h.prototype.stop=function(){this._core.is("rotating")&&(f.clearTimeout(this._timeout),this._core.leave("rotating"))
},h.prototype.pause=function(){this._core.is("rotating")&&(this._paused=!0)
},h.prototype.destroy=function(){var d,c;
this.stop();
for(d in this._handlers){this._core.$element.off(d,this._handlers[d])
}for(c in Object.getOwnPropertyNames(this)){"function"!=typeof this[c]&&(this[c]=null)
}},g.fn.owlCarousel.Constructor.Plugins.autoplay=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this._core=a,this._initialized=!1,this._pages=[],this._controls={},this._templates=[],this.$element=this._core.$element,this._overrides={next:this._core.next,prev:this._core.prev,to:this._core.to},this._handlers={"prepared.owl.carousel":g.proxy(function(c){c.namespace&&this._core.settings.dotsData&&this._templates.push('<div class="'+this._core.settings.dotClass+'">'+g(c.content).find("[data-dot]").addBack("[data-dot]").attr("data-dot")+"</div>")
},this),"added.owl.carousel":g.proxy(function(b){b.namespace&&this._core.settings.dotsData&&this._templates.splice(b.position,0,this._templates.pop())
},this),"remove.owl.carousel":g.proxy(function(b){b.namespace&&this._core.settings.dotsData&&this._templates.splice(b.position,1)
},this),"changed.owl.carousel":g.proxy(function(b){b.namespace&&"position"==b.property.name&&this.draw()
},this),"initialized.owl.carousel":g.proxy(function(b){b.namespace&&!this._initialized&&(this._core.trigger("initialize",null,"navigation"),this.initialize(),this.update(),this.draw(),this._initialized=!0,this._core.trigger("initialized",null,"navigation"))
},this),"refreshed.owl.carousel":g.proxy(function(b){b.namespace&&this._initialized&&(this._core.trigger("refresh",null,"navigation"),this.update(),this.draw(),this._core.trigger("refreshed",null,"navigation"))
},this)},this._core.options=g.extend({},h.Defaults,this._core.options),this.$element.on(this._handlers)
};
h.Defaults={nav:!1,navText:["prev","next"],navSpeed:!1,navElement:"div",navContainer:!1,navContainerClass:"owl-nav",navClass:["owl-prev","owl-next"],slideBy:1,dotClass:"owl-dot",dotsClass:"owl-dots",dots:!0,dotsEach:!1,dotsData:!1,dotsSpeed:!1,dotsContainer:!1},h.prototype.initialize=function(){var a,d=this._core.settings;
this._controls.$relative=(d.navContainer?g(d.navContainer):g("<div>").addClass(d.navContainerClass).appendTo(this.$element)).addClass("disabled"),this._controls.$previous=g("<"+d.navElement+">").addClass(d.navClass[0]).html(d.navText[0]).prependTo(this._controls.$relative).on("click",g.proxy(function(b){this.prev(d.navSpeed)
},this)),this._controls.$next=g("<"+d.navElement+">").addClass(d.navClass[1]).html(d.navText[1]).appendTo(this._controls.$relative).on("click",g.proxy(function(b){this.next(d.navSpeed)
},this)),d.dotsData||(this._templates=[g("<div>").addClass(d.dotClass).append(g("<span>")).prop("outerHTML")]),this._controls.$absolute=(d.dotsContainer?g(d.dotsContainer):g("<div>").addClass(d.dotsClass).appendTo(this.$element)).addClass("disabled"),this._controls.$absolute.on("click","div",g.proxy(function(c){var e=g(c.target).parent().is(this._controls.$absolute)?g(c.target).index():g(c.target).parent().index();
c.preventDefault(),this.to(e,d.dotsSpeed)
},this));
for(a in this._overrides){this._core[a]=g.proxy(this[a],this)
}},h.prototype.destroy=function(){var k,e,m,l;
for(k in this._handlers){this.$element.off(k,this._handlers[k])
}for(e in this._controls){this._controls[e].remove()
}for(l in this.overides){this._core[l]=this._overrides[l]
}for(m in Object.getOwnPropertyNames(this)){"function"!=typeof this[m]&&(this[m]=null)
}},h.prototype.update=function(){var l,k,r,q=this._core.clones().length/2,p=q+this._core.items().length,o=this._core.maximum(!0),n=this._core.settings,m=n.center||n.autoWidth||n.dotsData?1:n.dotsEach||n.items;
if("page"!==n.slideBy&&(n.slideBy=Math.min(n.slideBy,n.items)),n.dots||"page"==n.slideBy){for(this._pages=[],l=q,k=0,r=0;
l<p;
l++){if(k>=m||0===k){if(this._pages.push({start:Math.min(o,l-q),end:l-q+m-1}),Math.min(o,l-q)===o){break
}k=0,++r
}k+=this._core.mergers(this._core.relative(l))
}}},h.prototype.draw=function(){var a,n=this._core.settings,m=this._core.items().length<=n.items,l=this._core.relative(this._core.current()),k=n.loop||n.rewind;
this._controls.$relative.toggleClass("disabled",!n.nav||m),n.nav&&(this._controls.$previous.toggleClass("disabled",!k&&l<=this._core.minimum(!0)),this._controls.$next.toggleClass("disabled",!k&&l>=this._core.maximum(!0))),this._controls.$absolute.toggleClass("disabled",!n.dots||m),n.dots&&(a=this._pages.length-this._controls.$absolute.children().length,n.dotsData&&0!==a?this._controls.$absolute.html(this._templates.join("")):a>0?this._controls.$absolute.append(new Array(a+1).join(this._templates[0])):a<0&&this._controls.$absolute.children().slice(a).remove(),this._controls.$absolute.find(".active").removeClass("active"),this._controls.$absolute.children().eq(g.inArray(this.current(),this._pages)).addClass("active"))
},h.prototype.onTrigger=function(a){var d=this._core.settings;
a.page={index:g.inArray(this.current(),this._pages),count:this._pages.length,size:d&&(d.center||d.autoWidth||d.dotsData?1:d.dotsEach||d.items)}
},h.prototype.current=function(){var a=this._core.relative(this._core.current());
return g.grep(this._pages,g.proxy(function(b,d){return b.start<=a&&b.end>=a
},this)).pop()
},h.prototype.getPosition=function(a){var m,l,k=this._core.settings;
return"page"==k.slideBy?(m=g.inArray(this.current(),this._pages),l=this._pages.length,a?++m:--m,m=this._pages[(m%l+l)%l].start):(m=this._core.relative(this._core.current()),l=this._core.items().length,a?m+=k.slideBy:m-=k.slideBy),m
},h.prototype.next=function(a){g.proxy(this._overrides.to,this._core)(this.getPosition(!0),a)
},h.prototype.prev=function(a){g.proxy(this._overrides.to,this._core)(this.getPosition(!1),a)
},h.prototype.to=function(a,m,l){var k;
!l&&this._pages.length?(k=this._pages.length,g.proxy(this._overrides.to,this._core)(this._pages[(a%k+k)%k].start,m)):g.proxy(this._overrides.to,this._core)(a,m)
},g.fn.owlCarousel.Constructor.Plugins.Navigation=h
}(window.Zepto||window.jQuery,window,document),function(g,f,j,i){var h=function(a){this._core=a,this._hashes={},this.$element=this._core.$element,this._handlers={"initialized.owl.carousel":g.proxy(function(b){b.namespace&&"URLHash"===this._core.settings.startPosition&&g(f).trigger("hashchange.owl.navigation")
},this),"prepared.owl.carousel":g.proxy(function(d){if(d.namespace){var e=g(d.content).find("[data-hash]").addBack("[data-hash]").attr("data-hash");
if(!e){return
}this._hashes[e]=d.content
}},this),"changed.owl.carousel":g.proxy(function(l){if(l.namespace&&"position"===l.property.name){var k=this._core.items(this._core.relative(this._core.current())),b=g.map(this._hashes,function(d,c){return d===k?c:null
}).join();
if(!b||f.location.hash.slice(1)===b){return
}f.location.hash=b
}},this)},this._core.options=g.extend({},h.Defaults,this._core.options),this.$element.on(this._handlers),g(f).on("hashchange.owl.navigation",g.proxy(function(b){var l=f.location.hash.substring(1),k=this._core.$stage.children(),d=this._hashes[l]&&k.index(this._hashes[l]);
d!==i&&d!==this._core.current()&&this._core.to(this._core.relative(d),!1,!0)
},this))
};
h.Defaults={URLhashListener:!1},h.prototype.destroy=function(){var b,a;
g(f).off("hashchange.owl.navigation");
for(b in this._handlers){this._core.$element.off(b,this._handlers[b])
}for(a in Object.getOwnPropertyNames(this)){"function"!=typeof this[a]&&(this[a]=null)
}},g.fn.owlCarousel.Constructor.Plugins.Hash=h
}(window.Zepto||window.jQuery,window,document),function(t,s,r,q){function p(a,h){var g=!1,d=a.charAt(0).toUpperCase()+a.slice(1);
return t.each((a+" "+m.join(d+" ")+d).split(" "),function(e,c){if(n[c]!==q){return g=!h||c,!1
}}),g
}function o(b){return p(b,!0)
}var n=t("<support>").get(0).style,m="Webkit Moz O ms".split(" "),l={transition:{end:{WebkitTransition:"webkitTransitionEnd",MozTransition:"transitionend",OTransition:"oTransitionEnd",transition:"transitionend"}},animation:{end:{WebkitAnimation:"webkitAnimationEnd",MozAnimation:"animationend",OAnimation:"oAnimationEnd",animation:"animationend"}}},k={csstransforms:function(){return !!p("transform")
},csstransforms3d:function(){return !!p("perspective")
},csstransitions:function(){return !!p("transition")
},cssanimations:function(){return !!p("animation")
}};
k.csstransitions()&&(t.support.transition=new String(o("transition")),t.support.transition.end=l.transition.end[t.support.transition]),k.cssanimations()&&(t.support.animation=new String(o("animation")),t.support.animation.end=l.animation.end[t.support.animation]),k.csstransforms()&&(t.support.transform=new String(o("transform")),t.support.transform3d=k.csstransforms3d())
}(window.Zepto||window.jQuery,window,document);
!function(b){"function"==typeof define&&define.amd?define(["jquery"],b):"object"==typeof exports?module.exports=b:b(jQuery)
}(function(v){function u(z){var y=z||window.event,x=n.call(arguments,1),w=0,k=0,i=0,f=0,e=0,d=0;
if(z=v.event.fix(y),z.type="mousewheel","detail" in y&&(i=-1*y.detail),"wheelDelta" in y&&(i=y.wheelDelta),"wheelDeltaY" in y&&(i=y.wheelDeltaY),"wheelDeltaX" in y&&(k=-1*y.wheelDeltaX),"axis" in y&&y.axis===y.HORIZONTAL_AXIS&&(k=-1*i,i=0),w=0===i?k:i,"deltaY" in y&&(i=-1*y.deltaY,w=i),"deltaX" in y&&(k=y.deltaX,0===i&&(w=-1*k)),0!==i||0!==k){if(1===y.deltaMode){var c=v.data(this,"mousewheel-line-height");
w*=c,i*=c,k*=c
}else{if(2===y.deltaMode){var a=v.data(this,"mousewheel-page-height");
w*=a,i*=a,k*=a
}}if(f=Math.max(Math.abs(i),Math.abs(k)),(!q||q>f)&&(q=f,s(y,f)&&(q/=40)),s(y,f)&&(w/=40,k/=40,i/=40),w=Math[w>=1?"floor":"ceil"](w/q),k=Math[k>=1?"floor":"ceil"](k/q),i=Math[i>=1?"floor":"ceil"](i/q),l.settings.normalizeOffset&&this.getBoundingClientRect){var A=this.getBoundingClientRect();
e=z.clientX-A.left,d=z.clientY-A.top
}return z.deltaX=k,z.deltaY=i,z.deltaFactor=q,z.offsetX=e,z.offsetY=d,z.deltaMode=0,x.unshift(z,w,k,i),r&&clearTimeout(r),r=setTimeout(t,200),(v.event.dispatch||v.event.handle).apply(this,x)
}}function t(){q=null
}function s(d,c){return l.settings.adjustOldDeltas&&"mousewheel"===d.type&&c%120===0
}var r,q,p=["wheel","mousewheel","DOMMouseScroll","MozMousePixelScroll"],o="onwheel" in document||document.documentMode>=9?["wheel"]:["mousewheel","DomMouseScroll","MozMousePixelScroll"],n=Array.prototype.slice;
if(v.event.fixHooks){for(var m=p.length;
m;
){v.event.fixHooks[p[--m]]=v.event.mouseHooks
}}var l=v.event.special.mousewheel={version:"3.1.12",setup:function(){if(this.addEventListener){for(var a=o.length;
a;
){this.addEventListener(o[--a],u,!1)
}}else{this.onmousewheel=u
}v.data(this,"mousewheel-line-height",l.getLineHeight(this)),v.data(this,"mousewheel-page-height",l.getPageHeight(this))
},teardown:function(){if(this.removeEventListener){for(var a=o.length;
a;
){this.removeEventListener(o[--a],u,!1)
}}else{this.onmousewheel=null
}v.removeData(this,"mousewheel-line-height"),v.removeData(this,"mousewheel-page-height")
},getLineHeight:function(a){var f=v(a),e=f["offsetParent" in v.fn?"offsetParent":"parent"]();
return e.length||(e=v("body")),parseInt(e.css("fontSize"),10)||parseInt(f.css("fontSize"),10)||16
},getPageHeight:function(a){return v(a).height()
},settings:{adjustOldDeltas:!0,normalizeOffset:!0}};
v.fn.extend({mousewheel:function(b){return b?this.bind("mousewheel",b):this.trigger("mousewheel")
},unmousewheel:function(b){return this.unbind("mousewheel",b)
}})
});
!function(b){"function"==typeof define&&define.amd?define(["jquery"],b):"object"==typeof exports?module.exports=b:b(jQuery)
}(function(v){function u(z){var y=z||window.event,x=n.call(arguments,1),w=0,k=0,i=0,f=0,e=0,d=0;
if(z=v.event.fix(y),z.type="mousewheel","detail" in y&&(i=-1*y.detail),"wheelDelta" in y&&(i=y.wheelDelta),"wheelDeltaY" in y&&(i=y.wheelDeltaY),"wheelDeltaX" in y&&(k=-1*y.wheelDeltaX),"axis" in y&&y.axis===y.HORIZONTAL_AXIS&&(k=-1*i,i=0),w=0===i?k:i,"deltaY" in y&&(i=-1*y.deltaY,w=i),"deltaX" in y&&(k=y.deltaX,0===i&&(w=-1*k)),0!==i||0!==k){if(1===y.deltaMode){var c=v.data(this,"mousewheel-line-height");
w*=c,i*=c,k*=c
}else{if(2===y.deltaMode){var a=v.data(this,"mousewheel-page-height");
w*=a,i*=a,k*=a
}}if(f=Math.max(Math.abs(i),Math.abs(k)),(!q||q>f)&&(q=f,s(y,f)&&(q/=40)),s(y,f)&&(w/=40,k/=40,i/=40),w=Math[w>=1?"floor":"ceil"](w/q),k=Math[k>=1?"floor":"ceil"](k/q),i=Math[i>=1?"floor":"ceil"](i/q),l.settings.normalizeOffset&&this.getBoundingClientRect){var A=this.getBoundingClientRect();
e=z.clientX-A.left,d=z.clientY-A.top
}return z.deltaX=k,z.deltaY=i,z.deltaFactor=q,z.offsetX=e,z.offsetY=d,z.deltaMode=0,x.unshift(z,w,k,i),r&&clearTimeout(r),r=setTimeout(t,200),(v.event.dispatch||v.event.handle).apply(this,x)
}}function t(){q=null
}function s(d,c){return l.settings.adjustOldDeltas&&"mousewheel"===d.type&&c%120===0
}var r,q,p=["wheel","mousewheel","DOMMouseScroll","MozMousePixelScroll"],o="onwheel" in document||document.documentMode>=9?["wheel"]:["mousewheel","DomMouseScroll","MozMousePixelScroll"],n=Array.prototype.slice;
if(v.event.fixHooks){for(var m=p.length;
m;
){v.event.fixHooks[p[--m]]=v.event.mouseHooks
}}var l=v.event.special.mousewheel={version:"3.1.12",setup:function(){if(this.addEventListener){for(var a=o.length;
a;
){this.addEventListener(o[--a],u,!1)
}}else{this.onmousewheel=u
}v.data(this,"mousewheel-line-height",l.getLineHeight(this)),v.data(this,"mousewheel-page-height",l.getPageHeight(this))
},teardown:function(){if(this.removeEventListener){for(var a=o.length;
a;
){this.removeEventListener(o[--a],u,!1)
}}else{this.onmousewheel=null
}v.removeData(this,"mousewheel-line-height"),v.removeData(this,"mousewheel-page-height")
},getLineHeight:function(a){var f=v(a),e=f["offsetParent" in v.fn?"offsetParent":"parent"]();
return e.length||(e=v("body")),parseInt(e.css("fontSize"),10)||parseInt(f.css("fontSize"),10)||16
},getPageHeight:function(a){return v(a).height()
},settings:{adjustOldDeltas:!0,normalizeOffset:!0}};
v.fn.extend({mousewheel:function(b){return b?this.bind("mousewheel",b):this.trigger("mousewheel")
},unmousewheel:function(b){return this.unbind("mousewheel",b)
}})
});
!function(a){"function"==typeof define&&define.amd?define(["jquery"],a):"undefined"!=typeof module&&module.exports?module.exports=a:a(jQuery,window,document)
}(function(a){!function(d){var e="function"==typeof define&&define.amd,b="undefined"!=typeof module&&module.exports,f="https:"==document.location.protocol?"https:":"http:",c="cdnjs.cloudflare.com/ajax/libs/jquery-mousewheel/3.1.13/jquery.mousewheel.min.js";
e||(b?require("jquery-mousewheel")(a):a.event.special.mousewheel||a("head").append(decodeURI("%3Cscript src="+f+"//"+c+"%3E%3C/script%3E"))),d()
}(function(){var aM,aR="mCustomScrollbar",a5="mCS",aS=".mCustomScrollbar",aX={setTop:0,setLeft:0,axis:"y",scrollbarPosition:"inside",scrollInertia:950,autoDraggerLength:!0,alwaysShowScrollbar:0,snapOffset:0,mouseWheel:{enable:!0,scrollAmount:"auto",axis:"y",deltaFactor:"auto",disableOver:["select","option","keygen","datalist","textarea"]},scrollButtons:{scrollType:"stepless",scrollAmount:"auto"},keyboard:{enable:!0,scrollType:"stepless",scrollAmount:"auto"},contentTouchScroll:25,documentTouchScroll:!0,advanced:{autoScrollOnFocus:"input,textarea,select,button,datalist,keygen,a[tabindex],area,object,[contenteditable='true']",updateOnContentResize:!0,updateOnImageLoad:"auto",autoUpdateTimeout:60},theme:"light",callbacks:{onTotalScrollOffset:0,onTotalScrollBackOffset:0,alwaysTriggerOffsets:!0}},aO=0,aU={},aN=window.attachEvent&&!window.addEventListener?1:0,a2=!1,a1=["mCSB_dragger_onDrag","mCSB_scrollTools_onDrag","mCS_img_loaded","mCS_disabled","mCS_destroyed","mCS_no_scrollbar","mCS-autoHide","mCS-dir-rtl","mCS_no_scrollbar_y","mCS_no_scrollbar_x","mCS_y_hidden","mCS_x_hidden","mCSB_draggerContainer","mCSB_buttonUp","mCSB_buttonDown","mCSB_buttonLeft","mCSB_buttonRight"],aL={init:function(b){var b=a.extend(!0,{},aX,b),f=a0.call(this);
if(b.live){var d=b.liveSelector||this.selector||aS,g=a(d);
if("off"===b.live){return void aT(d)
}aU[d]=setTimeout(function(){g.mCustomScrollbar(b),"once"===b.live&&g.length&&aT(d)
},500)
}else{aT(d)
}return b.setWidth=b.set_width?b.set_width:b.setWidth,b.setHeight=b.set_height?b.set_height:b.setHeight,b.axis=b.horizontalScroll?"x":aQ(b.axis),b.scrollInertia=b.scrollInertia>0&&b.scrollInertia<17?17:b.scrollInertia,"object"!=typeof b.mouseWheel&&1==b.mouseWheel&&(b.mouseWheel={enable:!0,scrollAmount:"auto",axis:"y",preventDefault:!1,deltaFactor:"auto",normalizeDelta:!1,invert:!1}),b.mouseWheel.scrollAmount=b.mouseWheelPixels?b.mouseWheelPixels:b.mouseWheel.scrollAmount,b.mouseWheel.normalizeDelta=b.advanced.normalizeMouseWheelDelta?b.advanced.normalizeMouseWheelDelta:b.mouseWheel.normalizeDelta,b.scrollButtons.scrollType=aZ(b.scrollButtons.scrollType),aY(b),a(f).each(function(){var m=a(this);
if(!m.data(a5)){m.data(a5,{idx:++aO,opt:b,scrollRatio:{y:null,x:null},overflowed:null,contentReset:{y:null,x:null},bindEvents:!1,tweenRunning:!1,sequential:{},langDir:m.css("direction"),cbOffsets:null,trigger:null,poll:{size:{o:0,n:0},img:{o:0,n:0},change:{o:0,n:0}}});
var q=m.data(a5),j=q.opt,h=m.data("mcs-axis"),k=m.data("mcs-scrollbar-position"),p=m.data("mcs-theme");
h&&(j.axis=h),k&&(j.scrollbarPosition=k),p&&(j.theme=p,aY(j)),aK.call(this),q&&j.callbacks.onCreate&&"function"==typeof j.callbacks.onCreate&&j.callbacks.onCreate.call(this),a("#mCSB_"+q.idx+"_container img:not(."+a1[2]+")").addClass(a1[2]),aL.update.call(null,m)
}})
},update:function(b,c){var d=b||a0.call(this);
return a(d).each(function(){var h=a(this);
if(h.data(a5)){var o=h.data(a5),g=o.opt,k=a("#mCSB_"+o.idx+"_container"),f=a("#mCSB_"+o.idx),j=[a("#mCSB_"+o.idx+"_dragger_vertical"),a("#mCSB_"+o.idx+"_dragger_horizontal")];
if(!k.length){return
}o.tweenRunning&&al(h),c&&o&&g.callbacks.onBeforeUpdate&&"function"==typeof g.callbacks.onBeforeUpdate&&g.callbacks.onBeforeUpdate.call(this),h.hasClass(a1[3])&&h.removeClass(a1[3]),h.hasClass(a1[4])&&h.removeClass(a1[4]),f.css("max-height","none"),f.height()!==h.height()&&f.css("max-height",h.height()),a6.call(this),"y"===g.axis||g.advanced.autoExpandHorizontalScroll||k.css("width",aI(k)),o.overflowed=aH.call(this),ap.call(this),g.autoDraggerLength&&aj.call(this),a3.call(this),ai.call(this);
var m=[Math.abs(k[0].offsetTop),Math.abs(k[0].offsetLeft)];
"x"!==g.axis&&(o.overflowed[0]?j[0].height()>j[0].parent().height()?aC.call(this):(aw(h,m[0].toString(),{dir:"y",dur:0,overwrite:"none"}),o.contentReset.y=null):(aC.call(this),"y"===g.axis?aV.call(this):"yx"===g.axis&&o.overflowed[1]&&aw(h,m[1].toString(),{dir:"x",dur:0,overwrite:"none"}))),"y"!==g.axis&&(o.overflowed[1]?j[1].width()>j[1].parent().width()?aC.call(this):(aw(h,m[1].toString(),{dir:"x",dur:0,overwrite:"none"}),o.contentReset.x=null):(aC.call(this),"x"===g.axis?aV.call(this):"yx"===g.axis&&o.overflowed[0]&&aw(h,m[0].toString(),{dir:"y",dur:0,overwrite:"none"}))),c&&o&&(2===c&&g.callbacks.onImageLoad&&"function"==typeof g.callbacks.onImageLoad?g.callbacks.onImageLoad.call(this):3===c&&g.callbacks.onSelectorChange&&"function"==typeof g.callbacks.onSelectorChange?g.callbacks.onSelectorChange.call(this):g.callbacks.onUpdate&&"function"==typeof g.callbacks.onUpdate&&g.callbacks.onUpdate.call(this)),ao.call(this)
}})
},scrollTo:function(b,c){if("undefined"!=typeof b&&null!=b){var d=a0.call(this);
return a(d).each(function(){var o=a(this);
if(o.data(a5)){var g=o.data(a5),j=g.opt,f={trigger:"external",scrollInertia:j.scrollInertia,scrollEasing:"mcsEaseInOut",moveDragger:!1,timeout:60,callbacks:!0,onStart:!0,onUpdate:!0,onComplete:!0},h=a.extend(!0,{},f,c),m=ab.call(this,b),k=h.scrollInertia>0&&h.scrollInertia<17?17:h.scrollInertia;
m[0]=ac.call(this,m[0],"y"),m[1]=ac.call(this,m[1],"x"),h.moveDragger&&(m[0]*=g.scrollRatio.y,m[1]*=g.scrollRatio.x),h.dur=e()?0:k,setTimeout(function(){null!==m[0]&&"undefined"!=typeof m[0]&&"x"!==j.axis&&g.overflowed[0]&&(h.dir="y",h.overwrite="all",aw(o,m[0].toString(),h)),null!==m[1]&&"undefined"!=typeof m[1]&&"y"!==j.axis&&g.overflowed[1]&&(h.dir="x",h.overwrite="none",aw(o,m[1].toString(),h))
},h.timeout)
}})
}},stop:function(){var b=a0.call(this);
return a(b).each(function(){var c=a(this);
c.data(a5)&&al(c)
})
},disable:function(b){var c=a0.call(this);
return a(c).each(function(){var d=a(this);
if(d.data(a5)){d.data(a5);
ao.call(this,"remove"),aV.call(this),b&&aC.call(this),ap.call(this,!0),d.addClass(a1[3])
}})
},destroy:function(){var b=a0.call(this);
return a(b).each(function(){var k=a(this);
if(k.data(a5)){var f=k.data(a5),h=f.opt,d=a("#mCSB_"+f.idx),g=a("#mCSB_"+f.idx+"_container"),j=a(".mCSB_"+f.idx+"_scrollbar");
h.live&&aT(h.liveSelector||a(b).selector),ao.call(this,"remove"),aV.call(this),aC.call(this),k.removeData(a5),aE(this,"mcs"),j.remove(),g.find("img."+a1[2]).removeClass(a1[2]),d.replaceWith(g.contents()),k.removeClass(aR+" _"+a5+"_"+f.idx+" "+a1[6]+" "+a1[7]+" "+a1[5]+" "+a1[3]).addClass(a1[4])
}})
}},a0=function(){return"object"!=typeof a(this)||a(this).length<1?aS:this
},aY=function(d){var g=["rounded","rounded-dark","rounded-dots","rounded-dots-dark"],b=["rounded-dots","rounded-dots-dark","3d","3d-dark","3d-thick","3d-thick-dark","inset","inset-dark","inset-2","inset-2-dark","inset-3","inset-3-dark"],h=["minimal","minimal-dark"],c=["minimal","minimal-dark"],f=["minimal","minimal-dark"];
d.autoDraggerLength=a.inArray(d.theme,g)>-1?!1:d.autoDraggerLength,d.autoExpandScrollbar=a.inArray(d.theme,b)>-1?!1:d.autoExpandScrollbar,d.scrollButtons.enable=a.inArray(d.theme,h)>-1?!1:d.scrollButtons.enable,d.autoHideScrollbar=a.inArray(d.theme,c)>-1?!0:d.autoHideScrollbar,d.scrollbarPosition=a.inArray(d.theme,f)>-1?"outside":d.scrollbarPosition
},aT=function(b){aU[b]&&(clearTimeout(aU[b]),aE(aU,b))
},aQ=function(b){return"yx"===b||"xy"===b||"auto"===b?"yx":"x"===b||"horizontal"===b?"x":"y"
},aZ=function(b){return"stepped"===b||"pixels"===b||"step"===b||"click"===b?"stepped":"stepless"
},aK=function(){var A=a(this),j=A.data(a5),q=j.opt,b=q.autoExpandScrollbar?" "+a1[1]+"_expand":"",o=["<div id='mCSB_"+j.idx+"_scrollbar_vertical' class='mCSB_scrollTools mCSB_"+j.idx+"_scrollbar mCS-"+q.theme+" mCSB_scrollTools_vertical"+b+"'><div class='"+a1[12]+"'><div id='mCSB_"+j.idx+"_dragger_vertical' class='mCSB_dragger' style='position:absolute;'><div class='mCSB_dragger_bar' /></div><div class='mCSB_draggerRail' /></div></div>","<div id='mCSB_"+j.idx+"_scrollbar_horizontal' class='mCSB_scrollTools mCSB_"+j.idx+"_scrollbar mCS-"+q.theme+" mCSB_scrollTools_horizontal"+b+"'><div class='"+a1[12]+"'><div id='mCSB_"+j.idx+"_dragger_horizontal' class='mCSB_dragger' style='position:absolute;'><div class='mCSB_dragger_bar' /></div><div class='mCSB_draggerRail' /></div></div>"],B="yx"===q.axis?"mCSB_vertical_horizontal":"x"===q.axis?"mCSB_horizontal":"mCSB_vertical",y="yx"===q.axis?o[0]+o[1]:"x"===q.axis?o[1]:o[0],z="yx"===q.axis?"<div id='mCSB_"+j.idx+"_container_wrapper' class='mCSB_container_wrapper' />":"",x=q.autoHideScrollbar?" "+a1[6]:"",v="x"!==q.axis&&"rtl"===j.langDir?" "+a1[7]:"";
q.setWidth&&A.css("width",q.setWidth),q.setHeight&&A.css("height",q.setHeight),q.setLeft="y"!==q.axis&&"rtl"===j.langDir?"989999px":q.setLeft,A.addClass(aR+" _"+a5+"_"+j.idx+x+v).wrapInner("<div id='mCSB_"+j.idx+"' class='mCustomScrollBox mCS-"+q.theme+" "+B+"'><div id='mCSB_"+j.idx+"_container' class='mCSB_container' style='position:relative; top:"+q.setTop+"; left:"+q.setLeft+";' dir='"+j.langDir+"' /></div>");
var k=a("#mCSB_"+j.idx),d=a("#mCSB_"+j.idx+"_container");
"y"===q.axis||q.advanced.autoExpandHorizontalScroll||d.css("width",aI(d)),"outside"===q.scrollbarPosition?("static"===A.css("position")&&A.css("position","relative"),A.css("overflow","visible"),k.addClass("mCSB_outside").after(y)):(k.addClass("mCSB_inside").append(y),d.wrap(z)),aJ.call(this);
var w=[a("#mCSB_"+j.idx+"_dragger_vertical"),a("#mCSB_"+j.idx+"_dragger_horizontal")];
w[0].css("min-height",w[0].height()),w[1].css("min-width",w[1].width())
},aI=function(c){var d=[c[0].scrollWidth,Math.max.apply(Math,c.children().map(function(){return a(this).outerWidth(!0)
}).get())],b=c.parent().width();
return d[0]>b?d[0]:d[1]>b?d[1]:"100%"
},a6=function(){var c=a(this),f=c.data(a5),g=f.opt,b=a("#mCSB_"+f.idx+"_container");
if(g.advanced.autoExpandHorizontalScroll&&"y"!==g.axis){b.css({width:"auto","min-width":0,"overflow-x":"scroll"});
var d=Math.ceil(b[0].scrollWidth);
3===g.advanced.autoExpandHorizontalScroll||2!==g.advanced.autoExpandHorizontalScroll&&d>b.parent().width()?b.css({width:d,"min-width":"100%","overflow-x":"inherit"}):b.css({"overflow-x":"inherit",position:"absolute"}).wrap("<div class='mCSB_h_wrapper' style='position:relative; left:0; width:999999px;' />").css({width:Math.ceil(b[0].getBoundingClientRect().right+0.4)-Math.floor(b[0].getBoundingClientRect().left),"min-width":"100%",position:"relative"}).unwrap()
}},aJ=function(){var d=a(this),h=d.data(a5),j=h.opt,c=a(".mCSB_"+h.idx+"_scrollbar:first"),g=aF(j.scrollButtons.tabindex)?"tabindex='"+j.scrollButtons.tabindex+"'":"",b=["<a href='#' class='"+a1[13]+"' "+g+" />","<a href='#' class='"+a1[14]+"' "+g+" />","<a href='#' class='"+a1[15]+"' "+g+" />","<a href='#' class='"+a1[16]+"' "+g+" />"],f=["x"===j.axis?b[2]:b[0],"x"===j.axis?b[3]:b[1],b[2],b[3]];
j.scrollButtons.enable&&c.prepend(f[0]).append(f[1]).next(".mCSB_scrollTools").prepend(f[2]).append(f[3])
},aj=function(){var q=a(this),f=q.data(a5),g=a("#mCSB_"+f.idx),j=a("#mCSB_"+f.idx+"_container"),b=[a("#mCSB_"+f.idx+"_dragger_vertical"),a("#mCSB_"+f.idx+"_dragger_horizontal")],h=[g.height()/j.outerHeight(!1),g.width()/j.outerWidth(!1)],m=[parseInt(b[0].css("min-height")),Math.round(h[0]*b[0].parent().height()),parseInt(b[1].css("min-width")),Math.round(h[1]*b[1].parent().width())],k=aN&&m[1]<m[0]?m[0]:m[1],p=aN&&m[3]<m[2]?m[2]:m[3];
b[0].css({height:k,"max-height":b[0].parent().height()-10}).find(".mCSB_dragger_bar").css({"line-height":m[0]+"px"}),b[1].css({width:p,"max-width":b[1].parent().width()-10})
},a3=function(){var d=a(this),h=d.data(a5),j=a("#mCSB_"+h.idx),c=a("#mCSB_"+h.idx+"_container"),g=[a("#mCSB_"+h.idx+"_dragger_vertical"),a("#mCSB_"+h.idx+"_dragger_horizontal")],b=[c.outerHeight(!1)-j.height(),c.outerWidth(!1)-j.width()],f=[b[0]/(g[0].parent().height()-g[0].height()),b[1]/(g[1].parent().width()-g[1].width())];
h.scrollRatio={y:f[0],x:f[1]}
},aB=function(d,c,f){var b=f?a1[0]+"_expanded":"",g=d.closest(".mCSB_scrollTools");
"active"===c?(d.toggleClass(a1[0]+" "+b),g.toggleClass(a1[1]),d[0]._draggable=d[0]._draggable?0:1):d[0]._draggable||("hide"===c?(d.removeClass(a1[0]),g.removeClass(a1[1])):(d.addClass(a1[0]),g.addClass(a1[1])))
},aH=function(){var f=a(this),j=f.data(a5),m=a("#mCSB_"+j.idx),d=a("#mCSB_"+j.idx+"_container"),h=null==j.overflowed?d.height():d.outerHeight(!1),b=null==j.overflowed?d.width():d.outerWidth(!1),g=d[0].scrollHeight,k=d[0].scrollWidth;
return g>h&&(h=g),k>b&&(b=k),[h>m.height(),b>m.width()]
},aC=function(){var d=a(this),h=d.data(a5),j=h.opt,c=a("#mCSB_"+h.idx),g=a("#mCSB_"+h.idx+"_container"),b=[a("#mCSB_"+h.idx+"_dragger_vertical"),a("#mCSB_"+h.idx+"_dragger_horizontal")];
if(al(d),("x"!==j.axis&&!h.overflowed[0]||"y"===j.axis&&h.overflowed[0])&&(b[0].add(g).css("top",0),aw(d,"_resetY")),"y"!==j.axis&&!h.overflowed[1]||"x"===j.axis&&h.overflowed[1]){var f=dx=0;
"rtl"===h.langDir&&(f=c.width()-g.outerWidth(!1),dx=Math.abs(f/h.scrollRatio.x)),g.css("left",f),b[1].css("left",dx),aw(d,"_resetX")
}},ai=function(){function c(){d=setTimeout(function(){a.event.special.mousewheel?(clearTimeout(d),af.call(f[0])):c()
},100)
}var f=a(this),g=f.data(a5),b=g.opt;
if(!g.bindEvents){if(au.call(this),b.contentTouchScroll&&aA.call(this),ay.call(this),b.mouseWheel.enable){var d;
c()
}am.call(this),ah.call(this),b.advanced.autoScrollOnFocus&&av.call(this),b.scrollButtons.enable&&ax.call(this),b.keyboard.enable&&aP.call(this),g.bindEvents=!0
}},aV=function(){var d=a(this),h=d.data(a5),j=h.opt,c=a5+"_"+h.idx,g=".mCSB_"+h.idx+"_scrollbar",b=a("#mCSB_"+h.idx+",#mCSB_"+h.idx+"_container,#mCSB_"+h.idx+"_container_wrapper,"+g+" ."+a1[12]+",#mCSB_"+h.idx+"_dragger_vertical,#mCSB_"+h.idx+"_dragger_horizontal,"+g+">a"),f=a("#mCSB_"+h.idx+"_container");
j.advanced.releaseDraggableSelectors&&b.add(a(j.advanced.releaseDraggableSelectors)),j.advanced.extraDraggableSelectors&&b.add(a(j.advanced.extraDraggableSelectors)),h.bindEvents&&(a(document).add(a(!aD()||top.document)).unbind("."+c),b.each(function(){a(this).unbind("."+c)
}),clearTimeout(d[0]._focusTimeout),aE(d[0],"_focusTimeout"),clearTimeout(h.sequential.step),aE(h.sequential,"step"),clearTimeout(f[0].onCompleteTimeout),aE(f[0],"onCompleteTimeout"),h.bindEvents=!1)
},ap=function(f){var j=a(this),m=j.data(a5),d=m.opt,h=a("#mCSB_"+m.idx+"_container_wrapper"),b=h.length?h:a("#mCSB_"+m.idx+"_container"),g=[a("#mCSB_"+m.idx+"_scrollbar_vertical"),a("#mCSB_"+m.idx+"_scrollbar_horizontal")],k=[g[0].find(".mCSB_dragger"),g[1].find(".mCSB_dragger")];
"x"!==d.axis&&(m.overflowed[0]&&!f?(g[0].add(k[0]).add(g[0].children("a")).css("display","block"),b.removeClass(a1[8]+" "+a1[10])):(d.alwaysShowScrollbar?(2!==d.alwaysShowScrollbar&&k[0].css("display","none"),b.removeClass(a1[10])):(g[0].css("display","none"),b.addClass(a1[10])),b.addClass(a1[8]))),"y"!==d.axis&&(m.overflowed[1]&&!f?(g[1].add(k[1]).add(g[1].children("a")).css("display","block"),b.removeClass(a1[9]+" "+a1[11])):(d.alwaysShowScrollbar?(2!==d.alwaysShowScrollbar&&k[1].css("display","none"),b.removeClass(a1[11])):(g[1].css("display","none"),b.addClass(a1[11])),b.addClass(a1[9]))),m.overflowed[0]||m.overflowed[1]?j.removeClass(a1[5]):j.addClass(a1[5])
},an=function(d){var g=d.type,b=d.target.ownerDocument!==document&&null!==frameElement?[a(frameElement).offset().top,a(frameElement).offset().left]:null,h=aD()&&d.target.ownerDocument!==top.document&&null!==frameElement?[a(d.view.frameElement).offset().top,a(d.view.frameElement).offset().left]:[0,0];
switch(g){case"pointerdown":case"MSPointerDown":case"pointermove":case"MSPointerMove":case"pointerup":case"MSPointerUp":return b?[d.originalEvent.pageY-b[0]+h[0],d.originalEvent.pageX-b[1]+h[1],!1]:[d.originalEvent.pageY,d.originalEvent.pageX,!1];
case"touchstart":case"touchmove":case"touchend":var c=d.originalEvent.touches[0]||d.originalEvent.changedTouches[0],f=d.originalEvent.touches.length||d.originalEvent.changedTouches.length;
return d.target.ownerDocument!==document?[c.screenY,c.screenX,f>1]:[c.pageY,c.pageX,f>1];
default:return b?[d.pageY-b[0]+h[0],d.pageX-b[1]+h[1],!1]:[d.pageY,d.pageX,!1]
}},au=function(){function B(l,g,d,m){if(w[0].idleTimer=z.scrollInertia<233?250:0,j.attr("id")===y[1]){var f="x",h=(j[0].offsetLeft-g+m)*s.scrollRatio.x
}else{var f="y",h=(j[0].offsetTop-l+d)*s.scrollRatio.y
}aw(b,h.toString(),{dir:f,drag:!0})
}var j,k,v,b=a(this),s=b.data(a5),z=s.opt,A=a5+"_"+s.idx,y=["mCSB_"+s.idx+"_dragger_vertical","mCSB_"+s.idx+"_dragger_horizontal"],w=a("#mCSB_"+s.idx+"_container"),q=a("#"+y[0]+",#"+y[1]),c=z.advanced.releaseDraggableSelectors?q.add(a(z.advanced.releaseDraggableSelectors)):q,x=z.advanced.extraDraggableSelectors?a(!aD()||top.document).add(a(z.advanced.extraDraggableSelectors)):a(!aD()||top.document);
q.bind("contextmenu."+A,function(d){d.preventDefault()
}).bind("mousedown."+A+" touchstart."+A+" pointerdown."+A+" MSPointerDown."+A,function(n){if(n.stopImmediatePropagation(),n.preventDefault(),a4(n)){a2=!0,aN&&(document.onselectstart=function(){return !1
}),aq.call(w,!1),al(b),j=a(this);
var h=j.offset(),g=an(n)[0]-h.top,i=an(n)[1]-h.left,o=j.height()+h.top,d=j.width()+h.left;
o>g&&g>0&&d>i&&i>0&&(k=g,v=i),aB(j,"active",z.autoExpandScrollbar)
}}).bind("touchmove."+A,function(h){h.stopImmediatePropagation(),h.preventDefault();
var f=j.offset(),g=an(h)[0]-f.top,d=an(h)[1]-f.left;
B(k,v,g,d)
}),a(document).add(x).bind("mousemove."+A+" pointermove."+A+" MSPointerMove."+A,function(h){if(j){var f=j.offset(),g=an(h)[0]-f.top,d=an(h)[1]-f.left;
if(k===g&&v===d){return
}B(k,v,g,d)
}}).add(c).bind("mouseup."+A+" touchend."+A+" pointerup."+A+" MSPointerUp."+A,function(){j&&(aB(j,"active",z.autoExpandScrollbar),j=null),a2=!1,aN&&(document.onselectstart=null),aq.call(w,!0)
})
},aA=function(){function bd(b){if(!ad(b)||a2||an(b)[2]){return void (aM=0)
}aM=1,bn=0,a9=0,bm=1,N.removeClass("mCS_touch_action");
var d=O.offset();
Z=an(b)[0]-d.top,bl=an(b)[1]-d.left,K=[an(b)[0],an(b)[1]]
}function be(p){if(ad(p)&&!a2&&!an(p)[2]&&(q.documentTouchScroll||p.preventDefault(),p.stopImmediatePropagation(),(!a9||bn)&&bm)){bk=ar();
var u=G.offset(),d=an(p)[0]-u.top,s=an(p)[1]-u.left,f="mcsLinearOut";
if(ae.push(d),c.push(s),K[2]=Math.abs(an(p)[0]-K[0]),K[3]=Math.abs(an(p)[1]-K[1]),bb.overflowed[0]){var k=a7[0].parent().height()-a7[0].height(),b=Z-d>0&&d-Z>-(k*bb.scrollRatio.y)&&(2*K[3]<K[2]||"yx"===q.axis)
}if(bb.overflowed[1]){var g=a7[1].parent().width()-a7[1].width(),m=bl-s>0&&s-bl>-(g*bb.scrollRatio.x)&&(2*K[2]<K[3]||"yx"===q.axis)
}b||m?(j||p.preventDefault(),bn=1):(a9=1,N.addClass("mCS_touch_action")),j&&p.preventDefault(),X="yx"===q.axis?[Z-d,bl-s]:"x"===q.axis?[null,bl-s]:[Z-d,null],O[0].idleTimer=250,bb.overflowed[0]&&a8(X[0],A,f,"y","all",!0),bb.overflowed[1]&&a8(X[1],A,f,"x",J,!0)
}}function bi(b){if(!ad(b)||a2||an(b)[2]){return void (aM=0)
}aM=1,b.stopImmediatePropagation(),al(N),bc=ar();
var d=G.offset();
bj=an(b)[0]-d.top,bf=an(b)[1]-d.left,ae=[],c=[]
}function ba(l){if(ad(l)&&!a2&&!an(l)[2]){bm=0,l.stopImmediatePropagation(),bn=0,a9=0,Y=ar();
var v=G.offset(),d=an(l)[0]-v.top,p=an(l)[1]-v.left;
if(!(Y-bk>30)){bo=1000/(Y-bc);
var g="mcsEaseOut",h=2.5>bo,b=h?[ae[ae.length-2],c[c.length-2]]:[0,0];
Q=h?[d-b[0],p-b[1]]:[d-bj,p-bf];
var s=[Math.abs(Q[0]),Math.abs(Q[1])];
bo=h?[Math.abs(Q[0]/4),Math.abs(Q[1]/4)]:[bo,bo];
var k=[Math.abs(O[0].offsetTop)-Q[0]*bg(s[0]/bo[0],bo[0]),Math.abs(O[0].offsetLeft)-Q[1]*bg(s[1]/bo[1],bo[1])];
X="yx"===q.axis?[k[0],k[1]]:"x"===q.axis?[null,k[1]]:[k[0],null],t=[4*s[0]+q.scrollInertia,4*s[1]+q.scrollInertia];
var m=parseInt(q.contentTouchScroll)||0;
X[0]=s[0]>m?X[0]:0,X[1]=s[1]>m?X[1]:0,bb.overflowed[0]&&a8(X[0],t[0],g,"y",J,!1),bb.overflowed[1]&&a8(X[1],t[1],g,"x",J,!1)
}}}function bg(d,b){var f=[1.5*b,2*b,b/1.5,b/2];
return d>90?b>4?f[0]:f[3]:d>60?b>3?f[3]:f[2]:d>30?b>8?f[1]:b>6?f[0]:b>4?b:f[2]:b>8?b:f[3]
}function a8(g,f,h,b,k,d){g&&aw(N,g.toString(),{dur:f,scrollEasing:h,dir:b,overwrite:k,drag:d})
}var bm,Z,bl,bj,bf,bc,bk,Y,Q,bo,X,t,bn,a9,N=a(this),bb=N.data(a5),q=bb.opt,bh=a5+"_"+bb.idx,G=a("#mCSB_"+bb.idx),O=a("#mCSB_"+bb.idx+"_container"),a7=[a("#mCSB_"+bb.idx+"_dragger_vertical"),a("#mCSB_"+bb.idx+"_dragger_horizontal")],ae=[],c=[],A=0,J="yx"===q.axis?"none":"all",K=[],F=O.find("iframe"),V=["touchstart."+bh+" pointerdown."+bh+" MSPointerDown."+bh,"touchmove."+bh+" pointermove."+bh+" MSPointerMove."+bh,"touchend."+bh+" pointerup."+bh+" MSPointerUp."+bh],j=void 0!==document.body.style.touchAction&&""!==document.body.style.touchAction;
O.bind(V[0],function(b){bd(b)
}).bind(V[1],function(b){be(b)
}),G.bind(V[0],function(b){bi(b)
}).bind(V[2],function(b){ba(b)
}),F.length&&F.each(function(){a(this).bind("load",function(){aD(this)&&a(this.contentDocument||this.contentWindow.document).bind(V[0],function(b){bd(b),bi(b)
}).bind(V[1],function(b){be(b)
}).bind(V[2],function(b){ba(b)
})
})
})
},ay=function(){function c(){return window.getSelection?window.getSelection().toString():document.selection&&"Control"!=document.selection.type?document.selection.createRange().text:0
}function g(f,d,h){q.type=h&&k?"stepped":"stepless",q.scrollAmount=10,aW(b,f,d,"mcsLinearOut",h?60:null)
}var k,b=a(this),j=b.data(a5),v=j.opt,q=j.sequential,t=a5+"_"+j.idx,p=a("#mCSB_"+j.idx+"_container"),m=p.parent();
p.bind("mousedown."+t,function(){aM||k||(k=1,a2=!0)
}).add(document).bind("mousemove."+t,function(h){if(!aM&&k&&c()){var d=p.offset(),f=an(h)[0]-d.top+p[0].offsetTop,i=an(h)[1]-d.left+p[0].offsetLeft;
f>0&&f<m.height()&&i>0&&i<m.width()?q.step&&g("off",null,"stepped"):("x"!==v.axis&&j.overflowed[0]&&(0>f?g("on",38):f>m.height()&&g("on",40)),"y"!==v.axis&&j.overflowed[1]&&(0>i?g("on",37):i>m.width()&&g("on",39)))
}}).bind("mouseup."+t+" dragend."+t,function(){aM||(k&&(k=0,g("off",null)),a2=!1)
})
},af=function(){function g(z,w){if(al(k),!aG(k,z.target)){var c="auto"!==f.mouseWheel.deltaFactor?parseInt(f.mouseWheel.deltaFactor):aN&&z.deltaFactor<100?100:z.deltaFactor||100,s=f.scrollInertia;
if("x"===f.axis||"x"===f.mouseWheel.axis){var y="x",q=[Math.round(c*p.scrollRatio.x),parseInt(f.mouseWheel.scrollAmount)],n="auto"!==f.mouseWheel.scrollAmount?q[1]:q[0]>=b.width()?0.9*b.width():q[0],l=Math.abs(a("#mCSB_"+p.idx+"_container")[0].offsetLeft),i=m[1][0].offsetLeft,o=m[1].parent().width()-m[1].width(),x="y"===f.mouseWheel.axis?z.deltaY||w:z.deltaX
}else{var y="y",q=[Math.round(c*p.scrollRatio.y),parseInt(f.mouseWheel.scrollAmount)],n="auto"!==f.mouseWheel.scrollAmount?q[1]:q[0]>=b.height()?0.9*b.height():q[0],l=Math.abs(a("#mCSB_"+p.idx+"_container")[0].offsetTop),i=m[0][0].offsetTop,o=m[0].parent().height()-m[0].height(),x=z.deltaY||w
}"y"===y&&!p.overflowed[0]||"x"===y&&!p.overflowed[1]||((f.mouseWheel.invert||z.webkitDirectionInvertedFromDevice)&&(x=-x),f.mouseWheel.normalizeDelta&&(x=0>x?-1:1),(x>0&&0!==i||0>x&&i!==o||f.mouseWheel.preventDefault)&&(z.stopImmediatePropagation(),z.preventDefault()),z.deltaFactor<5&&!f.mouseWheel.normalizeDelta&&(n=z.deltaFactor,s=17),aw(k,(l-x*n).toString(),{dir:y,dur:s}))
}}if(a(this).data(a5)){var k=a(this),p=k.data(a5),f=p.opt,h=a5+"_"+p.idx,b=a("#mCSB_"+p.idx),m=[a("#mCSB_"+p.idx+"_dragger_vertical"),a("#mCSB_"+p.idx+"_dragger_horizontal")],j=a("#mCSB_"+p.idx+"_container").find("iframe");
j.length&&j.each(function(){a(this).bind("load",function(){aD(this)&&a(this.contentDocument||this.contentWindow.document).bind("mousewheel."+h,function(c,d){g(c,d)
})
})
}),b.bind("mousewheel."+h,function(c,d){g(c,d)
})
}},ak=new Object,aD=function(d){var g=!1,b=!1,h=null;
if(void 0===d?b="#empty":void 0!==a(d).attr("id")&&(b=a(d).attr("id")),b!==!1&&void 0!==ak[b]){return ak[b]
}if(d){try{var c=d.contentDocument||d.contentWindow.document;
h=c.body.innerHTML
}catch(f){}g=null!==h
}else{try{var c=top.document;
h=c.body.innerHTML
}catch(f){}g=null!==h
}return b!==!1&&(ak[b]=g),g
},aq=function(c){var b=this.find("iframe");
if(b.length){var d=c?"auto":"none";
b.css("pointer-events",d)
}},aG=function(c,f){var g=f.nodeName.toLowerCase(),b=c.data(a5).opt.mouseWheel.disableOver,d=["select","textarea"];
return a.inArray(g,b)>-1&&!(a.inArray(g,d)>-1&&!a(f).is(":focus"))
},am=function(){var d,h=a(this),j=h.data(a5),c=a5+"_"+j.idx,g=a("#mCSB_"+j.idx+"_container"),b=g.parent(),f=a(".mCSB_"+j.idx+"_scrollbar ."+a1[12]);
f.bind("mousedown."+c+" touchstart."+c+" pointerdown."+c+" MSPointerDown."+c,function(i){a2=!0,a(i.target).hasClass("mCSB_dragger")||(d=1)
}).bind("touchend."+c+" pointerup."+c+" MSPointerUp."+c,function(){a2=!1
}).bind("click."+c,function(k){if(d&&(d=0,a(k.target).hasClass(a1[12])||a(k.target).hasClass("mCSB_draggerRail"))){al(h);
var m=a(this),n=m.find(".mCSB_dragger");
if(m.parent(".mCSB_scrollTools_horizontal").length>0){if(!j.overflowed[1]){return
}var p="x",l=k.pageX>n.offset().left?-1:1,o=Math.abs(g[0].offsetLeft)-l*(0.9*b.width())
}else{if(!j.overflowed[0]){return
}var p="y",l=k.pageY>n.offset().top?-1:1,o=Math.abs(g[0].offsetTop)-l*(0.9*b.height())
}aw(h,o.toString(),{dir:p,scrollEasing:"mcsEaseInOut"})
}})
},av=function(){var d=a(this),g=d.data(a5),h=g.opt,c=a5+"_"+g.idx,f=a("#mCSB_"+g.idx+"_container"),b=f.parent();
f.bind("focusin."+c,function(){var l=a(document.activeElement),j=f.find(".mCustomScrollBox").length,k=0;
l.is(h.advanced.autoScrollOnFocus)&&(al(d),clearTimeout(d[0]._focusTimeout),d[0]._focusTimer=j?(k+17)*j:0,d[0]._focusTimeout=setTimeout(function(){var n=[az(l)[0],az(l)[1]],i=[f[0].offsetTop,f[0].offsetLeft],m=[i[0]+n[0]>=0&&i[0]+n[0]<b.height()-l.outerHeight(!1),i[1]+n[1]>=0&&i[0]+n[1]<b.width()-l.outerWidth(!1)],o="yx"!==h.axis||m[0]||m[1]?"all":"none";
"x"===h.axis||m[0]||aw(d,n[0].toString(),{dir:"y",scrollEasing:"mcsEaseInOut",overwrite:o,dur:k}),"y"===h.axis||m[1]||aw(d,n[1].toString(),{dir:"x",scrollEasing:"mcsEaseInOut",overwrite:o,dur:k})
},d[0]._focusTimer))
})
},ah=function(){var c=a(this),d=c.data(a5),f=a5+"_"+d.idx,b=a("#mCSB_"+d.idx+"_container").parent();
b.bind("scroll."+f,function(){0===b.scrollTop()&&0===b.scrollLeft()||a(".mCSB_"+d.idx+"_scrollbar").css("visibility","hidden")
})
},ax=function(){var d=a(this),h=d.data(a5),j=h.opt,c=h.sequential,g=a5+"_"+h.idx,b=".mCSB_"+h.idx+"_scrollbar",f=a(b+">a");
f.bind("contextmenu."+g,function(i){i.preventDefault()
}).bind("mousedown."+g+" touchstart."+g+" pointerdown."+g+" MSPointerDown."+g+" mouseup."+g+" touchend."+g+" pointerup."+g+" MSPointerUp."+g+" mouseout."+g+" pointerout."+g+" MSPointerOut."+g+" click."+g,function(k){function m(l,n){c.scrollAmount=j.scrollButtons.scrollAmount,aW(d,l,n)
}if(k.preventDefault(),a4(k)){var i=a(this).attr("class");
switch(c.type=j.scrollButtons.scrollType,k.type){case"mousedown":case"touchstart":case"pointerdown":case"MSPointerDown":if("stepped"===c.type){return
}a2=!0,h.tweenRunning=!1,m("on",i);
break;
case"mouseup":case"touchend":case"pointerup":case"MSPointerUp":case"mouseout":case"pointerout":case"MSPointerOut":if("stepped"===c.type){return
}a2=!1,c.dir&&m("off",i);
break;
case"click":if("stepped"!==c.type||h.tweenRunning){return
}m("on",i)
}}})
},aP=function(){function y(n){function i(h,f){b.type=m.keyboard.scrollType,b.scrollAmount=m.keyboard.scrollAmount,"stepped"===b.type&&j.tweenRunning||aW(g,h,f)
}switch(n.type){case"blur":j.tweenRunning&&b.dir&&i("off",null);
break;
case"keydown":case"keyup":var d=n.keyCode?n.keyCode:n.which,r="on";
if("x"!==m.axis&&(38===d||40===d)||"y"!==m.axis&&(37===d||39===d)){if((38===d||40===d)&&!j.overflowed[0]||(37===d||39===d)&&!j.overflowed[1]){return
}"keyup"===n.type&&(r="off"),a(document.activeElement).is(x)||(n.preventDefault(),n.stopImmediatePropagation(),i(r,d))
}else{if(33===d||34===d){if((j.overflowed[0]||j.overflowed[1])&&(n.preventDefault(),n.stopImmediatePropagation()),"keyup"===n.type){al(g);
var u=34===d?-1:1;
if("x"===m.axis||"yx"===m.axis&&j.overflowed[1]&&!j.overflowed[0]){var o="x",c=Math.abs(w[0].offsetLeft)-u*(0.9*v.width())
}else{var o="y",c=Math.abs(w[0].offsetTop)-u*(0.9*v.height())
}aw(g,c.toString(),{dir:o,scrollEasing:"mcsEaseInOut"})
}}else{if((35===d||36===d)&&!a(document.activeElement).is(x)&&((j.overflowed[0]||j.overflowed[1])&&(n.preventDefault(),n.stopImmediatePropagation()),"keyup"===n.type)){if("x"===m.axis||"yx"===m.axis&&j.overflowed[1]&&!j.overflowed[0]){var o="x",c=35===d?Math.abs(v.width()-w.outerWidth(!1)):0
}else{var o="y",c=35===d?Math.abs(v.height()-w.outerHeight(!1)):0
}aw(g,c.toString(),{dir:o,scrollEasing:"mcsEaseInOut"})
}}}}}var g=a(this),j=g.data(a5),m=j.opt,b=j.sequential,k=a5+"_"+j.idx,z=a("#mCSB_"+j.idx),w=a("#mCSB_"+j.idx+"_container"),v=w.parent(),x="input,textarea,select,datalist,keygen,[contenteditable='true']",q=w.find("iframe"),p=["blur."+k+" keydown."+k+" keyup."+k];
q.length&&q.each(function(){a(this).bind("load",function(){aD(this)&&a(this.contentDocument||this.contentWindow.document).bind(p[0],function(c){y(c)
})
})
}),z.attr("tabindex","0").bind(p[0],function(c){y(c)
})
},aW=function(C,j,k,w,b){function v(h){B.snapAmount&&(z.scrollAmount=B.snapAmount instanceof Array?"x"===z.dir[0]?B.snapAmount[1]:B.snapAmount[0]:B.snapAmount);
var c="stepped"!==z.type,p=b?b:h?c?d/1.5:y:1000/60,f=h?c?7.5:40:2.5,E=[Math.abs(x[0].offsetTop),Math.abs(x[0].offsetLeft)],i=[A.scrollRatio.y>10?10:A.scrollRatio.y,A.scrollRatio.x>10?10:A.scrollRatio.x],g="x"===z.dir[0]?E[1]+z.dir[1]*(i[1]*f):E[0]+z.dir[1]*(i[0]*f),u="x"===z.dir[0]?E[1]+z.dir[1]*parseInt(z.scrollAmount):E[0]+z.dir[1]*parseInt(z.scrollAmount),l="auto"!==z.scrollAmount?u:g,t=w?w:h?c?"mcsLinearOut":"mcsEaseInOut":"mcsLinear",r=!!h;
return h&&17>p&&(l="x"===z.dir[0]?E[1]:E[0]),aw(C,l.toString(),{dir:z.dir[0],scrollEasing:t,dur:p,onComplete:r}),h?void (z.dir=!1):(clearTimeout(z.step),void (z.step=setTimeout(function(){v()
},p)))
}function D(){clearTimeout(z.step),aE(z,"step"),al(C)
}var A=C.data(a5),B=A.opt,z=A.sequential,x=a("#mCSB_"+A.idx+"_container"),q="stepped"===z.type,d=B.scrollInertia<26?26:B.scrollInertia,y=B.scrollInertia<1?17:B.scrollInertia;
switch(j){case"on":if(z.dir=[k===a1[16]||k===a1[15]||39===k||37===k?"x":"y",k===a1[13]||k===a1[15]||38===k||37===k?-1:1],al(C),aF(k)&&"stepped"===z.type){return
}v(q);
break;
case"off":D(),(q||A.tweenRunning&&z.dir)&&v(!0)
}},ab=function(b){var c=a(this).data(a5).opt,d=[];
return"function"==typeof b&&(b=b()),b instanceof Array?d=b.length>1?[b[0],b[1]]:"x"===c.axis?[null,b[0]]:[b[0],null]:(d[0]=b.y?b.y:b.x||"x"===c.axis?null:b,d[1]=b.x?b.x:b.y||"y"===c.axis?null:b),"function"==typeof d[0]&&(d[0]=d[0]()),"function"==typeof d[1]&&(d[1]=d[1]()),d
},ac=function(A,j){if(null!=A&&"undefined"!=typeof A){var k=a(this),v=k.data(a5),b=v.opt,u=a("#mCSB_"+v.idx+"_container"),B=u.parent(),z=typeof A;
j||(j="x"===b.axis?"x":"y");
var y="x"===j?u.outerWidth(!1)-B.width():u.outerHeight(!1)-B.height(),x="x"===j?u[0].offsetLeft:u[0].offsetTop,w="x"===j?"left":"top";
switch(z){case"function":return A();
case"object":var q=A.jquery?A:a(A);
if(!q.length){return
}return"x"===j?az(q)[1]:az(q)[0];
case"string":case"number":if(aF(A)){return Math.abs(A)
}if(-1!==A.indexOf("%")){return Math.abs(y*parseInt(A)/100)
}if(-1!==A.indexOf("-=")){return Math.abs(x-parseInt(A.split("-=")[1]))
}if(-1!==A.indexOf("+=")){var g=x+parseInt(A.split("+=")[1]);
return g>=0?0:Math.abs(g)
}if(-1!==A.indexOf("px")&&aF(A.split("px")[0])){return Math.abs(A.split("px")[0])
}if("top"===A||"left"===A){return 0
}if("bottom"===A){return Math.abs(B.height()-u.outerHeight(!1))
}if("right"===A){return Math.abs(B.width()-u.outerWidth(!1))
}if("first"===A||"last"===A){var q=u.find(":"+A);
return"x"===j?az(q)[1]:az(q)[0]
}return a(A).length?"x"===j?az(a(A))[1]:az(a(A))[0]:(u.css(w,A),void aL.update.call(null,k[0]))
}}},ao=function(p){function d(){return clearTimeout(k[0].autoUpdate),0===h.parents("html").length?void (h=null):void (k[0].autoUpdate=setTimeout(function(){return m.advanced.updateOnSelectorChange&&(q.poll.change.n=j(),q.poll.change.n!==q.poll.change.o)?(q.poll.change.o=q.poll.change.n,void b(3)):m.advanced.updateOnContentResize&&(q.poll.size.n=h[0].scrollHeight+h[0].scrollWidth+k[0].offsetHeight+h[0].offsetHeight+h[0].offsetWidth,q.poll.size.n!==q.poll.size.o)?(q.poll.size.o=q.poll.size.n,void b(1)):!m.advanced.updateOnImageLoad||"auto"===m.advanced.updateOnImageLoad&&"y"===m.axis||(q.poll.img.n=k.find("img").length,q.poll.img.n===q.poll.img.o)?void ((m.advanced.updateOnSelectorChange||m.advanced.updateOnContentResize||m.advanced.updateOnImageLoad)&&d()):(q.poll.img.o=q.poll.img.n,void k.find("img").each(function(){g(this)
}))
},m.advanced.autoUpdateTimeout))
}function g(f){function i(o,n){return function(){return n.apply(o,arguments)
}
}function c(){this.onload=null,a(f).addClass(a1[2]),b(2)
}if(a(f).hasClass(a1[2])){return void b()
}var l=new Image;
l.onload=i(l,c),l.src=f.src
}function j(){m.advanced.updateOnSelectorChange===!0&&(m.advanced.updateOnSelectorChange="*");
var f=0,c=k.find(m.advanced.updateOnSelectorChange);
return m.advanced.updateOnSelectorChange&&c.length>0&&c.each(function(){f+=this.offsetHeight+this.offsetWidth
}),f
}function b(c){clearTimeout(k[0].autoUpdate),aL.update.call(null,h[0],c)
}var h=a(this),q=h.data(a5),m=q.opt,k=a("#mCSB_"+q.idx+"_container");
return p?(clearTimeout(k[0].autoUpdate),void aE(k[0],"autoUpdate")):void d()
},ag=function(c,b,d){return Math.round(c/b)*b-d
},al=function(b){var c=b.data(a5),d=a("#mCSB_"+c.idx+"_container,#mCSB_"+c.idx+"_container_wrapper,#mCSB_"+c.idx+"_dragger_vertical,#mCSB_"+c.idx+"_dragger_horizontal");
d.each(function(){aa.call(this)
})
},aw=function(F,K,L){function P(b){return G&&W.callbacks[b]&&"function"==typeof W.callbacks[b]
}function H(){return[W.callbacks.alwaysTriggerOffsets||C>=q[0]+z,W.callbacks.alwaysTriggerOffsets||-I>=C]
}function N(){var d=[Q[0].offsetTop,Q[0].offsetLeft],f=[A[0].offsetTop,A[0].offsetLeft],b=[Q.outerHeight(!1),Q.outerWidth(!1)],c=[U.height(),U.width()];
F[0].mcs={content:Q,top:d[0],left:d[1],draggerTop:f[0],draggerLeft:f[1],topPct:Math.round(100*Math.abs(d[0])/(Math.abs(b[0])-c[0])),leftPct:Math.round(100*Math.abs(d[1])/(Math.abs(b[1])-c[1])),direction:L.dir}
}var G=F.data(a5),W=G.opt,V={trigger:"internal",dir:"y",scrollEasing:"mcsEaseOut",drag:!1,dur:W.scrollInertia,overwrite:"all",callbacks:!0,onStart:!0,onUpdate:!0,onComplete:!0},L=a.extend(V,L),E=[L.dur,L.drag?0:L.dur],U=a("#mCSB_"+G.idx),Q=a("#mCSB_"+G.idx+"_container"),M=Q.parent(),J=W.callbacks.onTotalScrollOffset?ab.call(F,W.callbacks.onTotalScrollOffset):[0,0],R=W.callbacks.onTotalScrollBackOffset?ab.call(F,W.callbacks.onTotalScrollBackOffset):[0,0];
if(G.trigger=L.trigger,0===M.scrollTop()&&0===M.scrollLeft()||(a(".mCSB_"+G.idx+"_scrollbar").css("visibility","visible"),M.scrollTop(0).scrollLeft(0)),"_resetY"!==K||G.contentReset.y||(P("onOverflowYNone")&&W.callbacks.onOverflowYNone.call(F[0]),G.contentReset.y=1),"_resetX"!==K||G.contentReset.x||(P("onOverflowXNone")&&W.callbacks.onOverflowXNone.call(F[0]),G.contentReset.x=1),"_resetY"!==K&&"_resetX"!==K){if(!G.contentReset.y&&F[0].mcs||!G.overflowed[0]||(P("onOverflowY")&&W.callbacks.onOverflowY.call(F[0]),G.contentReset.x=null),!G.contentReset.x&&F[0].mcs||!G.overflowed[1]||(P("onOverflowX")&&W.callbacks.onOverflowX.call(F[0]),G.contentReset.x=null),W.snapAmount){var D=W.snapAmount instanceof Array?"x"===L.dir?W.snapAmount[1]:W.snapAmount[0]:W.snapAmount;
K=ag(K,D,W.snapOffset)
}switch(L.dir){case"x":var A=a("#mCSB_"+G.idx+"_dragger_horizontal"),Y="left",C=Q[0].offsetLeft,q=[U.width()-Q.outerWidth(!1),A.parent().width()-A.width()],X=[K,0===K?0:K/G.scrollRatio.x],z=J[1],I=R[1],j=z>0?z/G.scrollRatio.x:0,O=I>0?I/G.scrollRatio.x:0;
break;
case"y":var A=a("#mCSB_"+G.idx+"_dragger_vertical"),Y="top",C=Q[0].offsetTop,q=[U.height()-Q.outerHeight(!1),A.parent().height()-A.height()],X=[K,0===K?0:K/G.scrollRatio.y],z=J[0],I=R[0],j=z>0?z/G.scrollRatio.y:0,O=I>0?I/G.scrollRatio.y:0
}X[1]<0||0===X[0]&&0===X[1]?X=[0,0]:X[1]>=q[1]?X=[q[0],q[1]]:X[0]=-X[0],F[0].mcs||(N(),P("onInit")&&W.callbacks.onInit.call(F[0])),clearTimeout(Q[0].onCompleteTimeout),at(A[0],Y,Math.round(X[1]),E[1],L.scrollEasing),!G.tweenRunning&&(0===C&&X[0]>=0||C===q[0]&&X[0]<=q[0])||at(Q[0],Y,Math.round(X[0]),E[0],L.scrollEasing,L.overwrite,{onStart:function(){L.callbacks&&L.onStart&&!G.tweenRunning&&(P("onScrollStart")&&(N(),W.callbacks.onScrollStart.call(F[0])),G.tweenRunning=!0,aB(A),G.cbOffsets=H())
},onUpdate:function(){L.callbacks&&L.onUpdate&&P("whileScrolling")&&(N(),W.callbacks.whileScrolling.call(F[0]))
},onComplete:function(){if(L.callbacks&&L.onComplete){"yx"===W.axis&&clearTimeout(Q[0].onCompleteTimeout);
var b=Q[0].idleTimer||0;
Q[0].onCompleteTimeout=setTimeout(function(){P("onScroll")&&(N(),W.callbacks.onScroll.call(F[0])),P("onTotalScroll")&&X[1]>=q[1]-j&&G.cbOffsets[0]&&(N(),W.callbacks.onTotalScroll.call(F[0])),P("onTotalScrollBack")&&X[1]<=O&&G.cbOffsets[1]&&(N(),W.callbacks.onTotalScrollBack.call(F[0])),G.tweenRunning=!1,Q[0].idleTimer=0,aB(A,"hide")
},b)
}}})
}},at=function(M,A,E,Q,F,I,C){function H(){j.stop||(k||G.call(),k=ar()-y,B(),k>=j.time&&(j.time=k>j.time?k+L-(k-j.time):k+L-1,j.time<k+1&&(j.time=k+1)),j.time<Q?j.id=J(H):K.call())
}function B(){Q>0?(j.currVal=z(j.time,R,P,Q,F),q[A]=Math.round(j.currVal)+"px"):q[A]=E+"px",D.call()
}function O(){L=1000/60,j.time=k+L,J=window.requestAnimationFrame?window.requestAnimationFrame:function(b){return B(),setTimeout(b,0.01)
},j.id=J(H)
}function N(){null!=j.id&&(window.requestAnimationFrame?window.cancelAnimationFrame(j.id):clearTimeout(j.id),j.id=null)
}function z(g,d,h,b,l){switch(l){case"linear":case"mcsLinear":return h*g/b+d;
case"mcsLinearOut":return g/=b,g--,h*Math.sqrt(1-g*g)+d;
case"easeInOutSmooth":return g/=b/2,1>g?h/2*g*g+d:(g--,-h/2*(g*(g-2)-1)+d);
case"easeInOutStrong":return g/=b/2,1>g?h/2*Math.pow(2,10*(g-1))+d:(g--,h/2*(-Math.pow(2,-10*g)+2)+d);
case"easeInOut":case"mcsEaseInOut":return g/=b/2,1>g?h/2*g*g*g+d:(g-=2,h/2*(g*g*g+2)+d);
case"easeOutSmooth":return g/=b,g--,-h*(g*g*g*g-1)+d;
case"easeOutStrong":return h*(-Math.pow(2,-10*g/b)+1)+d;
case"easeOut":case"mcsEaseOut":default:var c=(g/=b)*g,f=c*g;
return d+h*(0.499999999999997*f*c+-2.5*c*c+5.5*f+-6.5*c+4*g)
}}M._mTween||(M._mTween={top:{},left:{}});
var L,J,C=C||{},G=C.onStart||function(){},D=C.onUpdate||function(){},K=C.onComplete||function(){},y=ar(),k=0,R=M.offsetTop,q=M.style,j=M._mTween[A];
"left"===A&&(R=M.offsetLeft);
var P=E-R;
j.stop=0,"none"!==I&&N(),O()
},ar=function(){return window.performance&&window.performance.now?window.performance.now():window.performance&&window.performance.webkitNow?window.performance.webkitNow():Date.now?Date.now():(new Date).getTime()
},aa=function(){var d=this;
d._mTween||(d._mTween={top:{},left:{}});
for(var c=["top","left"],f=0;
f<c.length;
f++){var b=c[f];
d._mTween[b].id&&(window.requestAnimationFrame?window.cancelAnimationFrame(d._mTween[b].id):clearTimeout(d._mTween[b].id),d._mTween[b].id=null,d._mTween[b].stop=1)
}},aE=function(c,b){try{delete c[b]
}catch(d){c[b]=null
}},a4=function(b){return !(b.which&&1!==b.which)
},ad=function(c){var b=c.originalEvent.pointerType;
return !(b&&"touch"!==b&&2!==b)
},aF=function(b){return !isNaN(parseFloat(b))&&isFinite(b)
},az=function(c){var b=c.parents(".mCSB_container");
return[c.offset().top-b.offset().top,c.offset().left-b.offset().left]
},e=function(){function c(){var f=["webkit","moz","ms","o"];
if("hidden" in document){return"hidden"
}for(var d=0;
d<f.length;
d++){if(f[d]+"Hidden" in document){return f[d]+"Hidden"
}}return null
}var b=c();
return b?document[b]:!1
};
a.fn[aR]=function(b){return aL[b]?aL[b].apply(this,Array.prototype.slice.call(arguments,1)):"object"!=typeof b&&b?void a.error("Method "+b+" does not exist"):aL.init.apply(this,arguments)
},a[aR]=function(b){return aL[b]?aL[b].apply(this,Array.prototype.slice.call(arguments,1)):"object"!=typeof b&&b?void a.error("Method "+b+" does not exist"):aL.init.apply(this,arguments)
},a[aR].defaults=aX,window[aR]=!0,a(window).bind("load",function(){a(aS)[aR](),a.extend(a.expr[":"],{mcsInView:a.expr[":"].mcsInView||function(d){var f,b,g=a(d),c=g.parents(".mCSB_container");
if(c.length){return f=c.parent(),b=[c[0].offsetTop,c[0].offsetLeft],b[0]+az(g)[0]>=0&&b[0]+az(g)[0]<f.height()-g.outerHeight(!1)&&b[1]+az(g)[1]>=0&&b[1]+az(g)[1]<f.width()-g.outerWidth(!1)
}},mcsInSight:a.expr[":"].mcsInSight||function(q,f,p){var g,j,b,h,u=a(q),m=u.parents(".mCSB_container"),k="exact"===p[3]?[[1,0],[1,0]]:[[0.9,0.1],[0.6,0.4]];
if(m.length){return g=[u.outerHeight(!1),u.outerWidth(!1)],b=[m[0].offsetTop+az(u)[0],m[0].offsetLeft+az(u)[1]],j=[m.parent()[0].offsetHeight,m.parent()[0].offsetWidth],h=[g[0]<j[0]?k[0]:k[1],g[1]<j[1]?k[0]:k[1]],b[0]-j[0]*h[0][0]<0&&b[0]+g[0]-j[0]*h[0][1]>=0&&b[1]-j[1]*h[1][0]<0&&b[1]+g[1]-j[1]*h[1][1]>=0
}},mcsOverflow:a.expr[":"].mcsOverflow||function(b){var c=a(b).data(a5);
if(c){return c.overflowed[0]||c.overflowed[1]
}}})
})
})
});