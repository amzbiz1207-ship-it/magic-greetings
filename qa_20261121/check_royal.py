import ast
from pathlib import Path
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parent.parent;s=(root/'app.py').read_text();tree=ast.parse(s)
asset=next(ast.get_source_segment(s,n) for n in tree.body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id','')=='LANDING_ASSETS') if any(isinstance(n,ast.AnnAssign) and getattr(n.target,'id','')=='LANDING_ASSETS' for n in tree.body) else next(ast.get_source_segment(s,n) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='LANDING_ASSETS' for t in n.targets))
prefix=s[:s.index('st.set_page_config(')].replace('ROOT = Path(__file__).resolve().parent',f'ROOT = Path({str(root)!r})')
prefix+='\n'+asset+'\n'+s[s.index('def render_original('):s.index('@st.cache_data(show_spinner=False)',s.index('def render_original('))]
harness=prefix+'''
from concurrent.futures import Future
class ControlledPool:
 def submit(self,*args,**kwargs):
  st.session_state['submitted_args']=args
  return Future()
if 'scene_executor' not in st.session_state:st.session_state['scene_executor']=ControlledPool()
photo=royal_template_image()
try:
 scene_editor(photo,'royal_fixture','royal','en','','','', 'fake-test-key',{'create':'Create card','preview':'Preview','download':'Download'})
except ScenePending:
 watch_scene_request(False)
'''
a=AppTest.from_string(harness,default_timeout=25).run();assert not a.exception,a.exception
assert len(a.checkbox)==1, [x.value for x in a.warning]
next(b for b in a.button if b.label=='Create card').click().run();assert not a.exception,a.exception
args=a.session_state['submitted_args'];assert args[2] is not None;assert 'Replace ONLY' in args[3]
assert args[2].getpixel((0,0))==0,'Background must be protected'
a.session_state['scene_pending_request']['future'].set_result(a.session_state['scene_source'].copy());a.run();assert not a.exception,a.exception
assert len(a.get('download_button'))==1
print('PASS: actual royal asset -> single face detection -> face-only mask -> reference photo -> completed portrait download')
b=AppTest.from_file(str(root/'app.py'),default_timeout=25).run();next(x for x in b.button if x.key=='quick_choose_royal').click().run();assert not b.exception,b.exception
assert next(x for x in b.button if x.key=='quick_choose_royal').label.startswith('✓')
assert next(x for x in b.button if x.key=='choose_royal').label=='✓ Selected'
assert len(b.get('file_uploader'))==1
print('PASS: royal quick selection and catalogue show selected state and uploader')
