from pathlib import Path
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parent.parent
source=(root/'app.py').read_text();prefix=source[:source.index('st.set_page_config(')]
prefix=prefix.replace('ROOT = Path(__file__).resolve().parent',f'ROOT = Path({str(root)!r})')
prefix += source[source.index('def render_original('):source.index('@st.cache_data(show_spinner=False)',source.index('def render_original('))]
harness=prefix+'''
from concurrent.futures import Future
class ControlledPool:
 def submit(self,*args,**kwargs):
  st.session_state['test_submissions']=st.session_state.get('test_submissions',0)+1
  return Future()
if 'scene_executor' not in st.session_state: st.session_state['scene_executor']=ControlledPool()
photo=Image.new('RGB',(1024,1024),'#bbbbaa')
digest='fixture_ornament_identity_v3'
if 'scene_digest' not in st.session_state:
 st.session_state.update(scene_digest=digest,scene_source=photo,scene_faces=[(300,180,150,180)],scene_issues=[],scene_saved={},scene_versions=[],scene_confirmed_faces=[],scene_original_faces=[])
try:
 scene_editor(photo,'fixture','ornament','en','Merry Christmas','Warm wishes','', 'fake-test-key',{'create':'Create card','preview':'Preview','download':'Download'})
'''
harness += source[source.index('except ScenePending:\n    watch_scene_request(lang == \"ru\")'):]
harness = harness.replace('try:\n scene_editor(', 'lang=\"en\";scene_mode=True\ntry:\n scene_editor(')
at=AppTest.from_string(harness,default_timeout=20).run();assert not at.exception,at.exception
at.button[0].click().run();assert not at.exception,at.exception
assert at.session_state['test_submissions']==1
assert len(at.info)>0
at.run();assert at.session_state['test_submissions']==1
at.session_state['scene_pending_request']['future'].set_result(__import__('PIL.Image',fromlist=['Image']).new('RGB',(1024,1024),'#bbbbaa'))
at.run();assert not at.exception,at.exception
assert len(at.session_state['scene_versions'])==1
assert len(at.get('download_button'))==1
assert at.session_state['test_submissions']==1
print('PASS: create -> pending -> rerun without duplicate -> completed -> actual greeting rendering -> PNG download')
at.text_area[0].set_value('Keep the original expression').run()
next(b for b in at.button if b.label=='Apply refinement').click().run()
assert not at.exception,at.exception
assert at.session_state['test_submissions']==2
at.run();assert at.session_state['test_submissions']==2
at.session_state['scene_pending_request']['future'].set_result(__import__('PIL.Image',fromlist=['Image']).new('RGB',(1024,1024),'#bbbbaa'))
at.run();assert not at.exception,at.exception
assert len(at.session_state['scene_versions'])==2
print('PASS: targeted refinement -> pending -> auto recovery -> second downloadable version')
for status,code,expected in [(401,'invalid_api_key','API key'),(429,'insufficient_quota','balance'),(503,'server_error','temporary error')]:
 import requests
 error_app=AppTest.from_string(harness,default_timeout=20).run()
 error_app.button[0].click().run()
 response=requests.Response();response.status_code=status;response._content=__import__('json').dumps({'error':{'code':code}}).encode()
 error_app.session_state['scene_pending_request']['future'].set_exception(requests.HTTPError(response=response))
 error_app.run();assert not error_app.exception,error_app.exception
 assert any(expected in e.value for e in error_app.error),error_app.error
 assert 'scene_resume' not in error_app.session_state
 print('PASS: HTTP',status,code,'-> readable error and released resume state')
full=AppTest.from_file(str(root/'app.py'),default_timeout=20).run();assert not full.exception,full.exception
next(b for b in full.button if b.key=='quick_choose_market').click().run();assert not full.exception,full.exception
assert len(full.get('file_uploader'))==1
print('PASS: full landing -> scene selection -> uploader')
