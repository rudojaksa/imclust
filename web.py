
CSS="""
html {
  background-color: #444051;
  font-family: 'Roboto Condensed', sans-serif; }
div {
  display: inline-block; }
div.box {
  vertical-align: middle;
  position: relative; }
h2 {
  color: white; }
"""

HTML="""<!DOCTYPE html>
<html>
<head>
<title>imclust.py</title>
<link rel="preconnect" href="https://fonts.gstatic.com">
<link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700&display=swap" rel="stylesheet">
<style type="text/css">
{CSS}
</style>
</head>
<body>
{BODY}
<br><br><br><br>
</body>
</html>
"""

def addimg(path,clas,title):
  s = ""
  s += f'<a href="{path}"><div class="box {clas}">'
  s += f'<div><img class={clas} src="{path}" title="{title}" height=128px></div>'
  s += f'</div></a>\n'
  return s

