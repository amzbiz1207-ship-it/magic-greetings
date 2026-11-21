import ast,base64,hashlib,logging,os,time,uuid,unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import Future,ThreadPoolExecutor
from unittest.mock import Mock,patch
import requests
from PIL import Image,ImageOps
source=(Path(__file__).resolve().parent.parent / 'app.py').read_text();tree=ast.parse(source)
ns=globals().copy();ns['st']=SimpleNamespace(session_state={});ns['IDENTITY_RULES']='identity '
def encode_image(im,fmt):
 b=BytesIO();im.save(b,format=fmt);return b.getvalue()
ns['encode_image']=encode_image
for node in tree.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('ScenePending','_request_scene','scene_request_pool','edit_scene'):
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual app functions>','exec'),ns)
class GenerationTests(unittest.TestCase):
 def setUp(self):
  ns['st'].session_state={};self.im=Image.new('RGB',(16,16),'red');self.mask=Image.new('L',(16,16),0)
 def tearDown(self):
  pool=ns['st'].session_state.get('scene_executor')
  if pool:pool.shutdown(wait=True)
 def response(self,im=None):
  im=im or self.im
  return SimpleNamespace(ok=True,json=lambda:{'data':[{'b64_json':base64.b64encode(encode_image(im,'PNG')).decode()}]})
 def call(self,mask=None):return ns['_request_scene'](self.im,mask,'test','fake-test-key')
 def test_01_success_decode(self):
  with patch.object(requests,'post',return_value=self.response()):self.assertEqual(self.call().size,(16,16))
 def test_02_protected_pixels(self):
  with patch.object(requests,'post',return_value=self.response(Image.new('RGB',(16,16),'blue'))):self.assertEqual(self.call(self.mask).getpixel((0,0)),(255,0,0))
 def test_03_dimensions_rejected(self):
  with patch.object(requests,'post',return_value=self.response(Image.new('RGB',(32,16)))):
   with self.assertRaises(ValueError):self.call()
 def test_04_bad_payload_rejected(self):
  with patch.object(requests,'post',return_value=SimpleNamespace(ok=True,json=lambda:{'data':[]})):
   with self.assertRaises(IndexError):self.call()
 def test_05_http_error_propagates(self):
  r=Mock(ok=False,status_code=401,headers={});r.json.return_value={'error':{'code':'invalid_api_key'}};r.raise_for_status.side_effect=requests.HTTPError('401')
  with patch.object(requests,'post',return_value=r):
   with self.assertRaises(requests.HTTPError):self.call()
 def test_06_network_timeout(self):
  with patch.object(requests,'post',side_effect=requests.Timeout):
   with self.assertRaises(requests.Timeout):self.call()
 def test_07_upload_contract(self):
  with patch.object(requests,'post',return_value=self.response()) as m:
   ns['_request_scene'](self.im,self.mask,'p','fake-test-key',reference=self.im,face_reference=self.im)
   kw=m.call_args.kwargs;self.assertEqual(len(kw['files']),4);self.assertEqual(kw['timeout'],(10,220));self.assertEqual(kw['data']['size'],'16x16')
 def pending(self):
  future=Future();pool=Mock();pool.submit.return_value=future;ns['st'].session_state['scene_executor']=pool
  return future,pool
 def test_08_pending_nonblocking_no_duplicate(self):
  f,p=self.pending()
  for _ in range(3):
   with self.assertRaises(ns['ScenePending']):ns['edit_scene'](self.im,None,'p','key')
  self.assertEqual(p.submit.call_count,1);self.assertIn('started',ns['st'].session_state['scene_pending_request'])
 def test_09_ready_result_recovered(self):
  f,p=self.pending()
  with self.assertRaises(ns['ScenePending']):ns['edit_scene'](self.im,None,'p','key')
  f.set_result(self.im);self.assertIs(ns['edit_scene'](self.im,None,'p','key'),self.im);self.assertNotIn('scene_pending_request',ns['st'].session_state)
 def test_10_failure_releases_pending(self):
  f,p=self.pending()
  with self.assertRaises(ns['ScenePending']):ns['edit_scene'](self.im,None,'p','key')
  f.set_exception(requests.Timeout())
  with self.assertRaises(requests.Timeout):ns['edit_scene'](self.im,None,'p','key')
  self.assertNotIn('scene_pending_request',ns['st'].session_state)
 def test_11_changed_input_does_not_duplicate_active(self):
  f,p=self.pending()
  for prompt in ('a','b'):
   with self.assertRaises(ns['ScenePending']):ns['edit_scene'](self.im,None,prompt,'key')
  self.assertEqual(p.submit.call_count,1)
 def test_12_session_pool_isolated(self):
  a=ns['scene_request_pool']();ns['st'].session_state={};b=ns['scene_request_pool']();self.assertIsNot(a,b);a.shutdown();b.shutdown()
if __name__=='__main__':unittest.main(verbosity=2)
