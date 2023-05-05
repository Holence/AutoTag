from utils import *
from DTPySide import *
from MainWindow import MainWindow

app=DTAPP(sys.argv)

app.setApplicationName("AutoTag")
app.setWindowIcon(DTIcon.HoloIcon1())
app.setAuthor("Holence")
app.setApplicationVersion("1.0.0.0")

session=MainWindow(app)

session.setSeed(114514)
session.setModel("BERT", "Store")
# session.setModel("BERT", "Generate")
# session.setModel("LSTM", "Store")
# session.setModel("LSTM", "Generate")

app.setMainSession(session)
app.run()