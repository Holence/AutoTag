from utils import *
from DTPySide import *
from MainWindow import MainWindow

app=DTAPP(sys.argv)

app.setApplicationName("AutoTag")
app.setWindowIcon(DTIcon.HoloIcon1())
app.setAuthor("Holence")
app.setApplicationVersion("1.0.0.0")
app.setLoginEnable(False)

session=MainWindow(app)
session.setModel("BERT")
app.setMainSession(session)

seed_torch(114514)

app.run()