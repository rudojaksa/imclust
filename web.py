
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
.bad {
  position: relative;
  overflow: hidden; }
.bad:before, .bad:after {
  position: absolute;
  content: '';
  background: red;
  display: block;
  width: 75%;
  height: 8px;
  -webkit-transform: rotate(-45deg);
  transform: rotate(-45deg);
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  margin: auto; }
.bad:after {
  -webkit-transform: rotate(45deg);    
  transform: rotate(45deg); }
"""

HTML="""<!DOCTYPE html>
<html>
<head>
<title>imclust</title>
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

def addimg(path,clas,title,bad):
  b = " bad" if bad else ""
  s = ""
  s += f'<a href="{path}"><div class="box {clas}{b}">'
  s += f'<div><img class={clas} src="{path}" title="{title}"></div>'
  s += f'</div></a>\n'
  return s

