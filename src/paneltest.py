#!/usr/bin/python

 # mobile.py

import wx

class Border(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, size=(300, 640))

 ###Set up the frame layout###
        borderPanel = wx.Panel(self, -1) # Make a new panel inside the frame \
        borderPanel.SetBackgroundColour('#ededed') #(using as border)

        panel = wx.Panel(borderPanel, -1) # Make a new panel inside the frame
        panel.SetBackgroundColour('BLACK') # Set 'panels' bgcolor

        topPan = wx.Panel(panel, -1) # Make a subpanel for the top of 'panel'
        topPan.SetBackgroundColour('WHITE') # Set 'topPans' bgcolor

        smallPan = wx.Panel(panel , -1) # Make a subpanel for the top of 'panel'
        smallPan.SetBackgroundColour('#4f5049') # Set 'smallPans' bgcolor


        bbox = wx.BoxSizer(wx.VERTICAL) # Make a new sizer for 'borderPanel'
        vbox = wx.BoxSizer(wx.VERTICAL) # Make a new sizer for 'panel'

        bbox.Add(panel, 1, wx.EXPAND | wx.ALL, 20)# Add panel to border panel
        vbox.Add(topPan, 5, wx.EXPAND | wx.ALL, 10) # Add both subpanels to the new sizer \
        vbox.Add(smallPan, 4, wx.EXPAND | wx.ALL, 10) # in order

        borderPanel.SetSizer(bbox) # set 'panel' inside 'borderPanel'
        panel.SetSizer(vbox) # set the sizer containing 'topPan' and 'smallPan' \
 # inside 'panel

 ### Add the buttons to smallPan###
        button1=wx.Button(smallPan,label="1",pos=(30,30),size=(50,30))
        button2=wx.Button(smallPan,label="2",pos=(90,30),size=(50,30))
        button3=wx.Button(smallPan,label="3",pos=(150,30),size=(50,30))
        button4=wx.Button(smallPan,label="4",pos=(30,70),size=(50,30))
        button5=wx.Button(smallPan,label="5",pos=(90,70),size=(50,30))
        button6=wx.Button(smallPan,label="6",pos=(150,70),size=(50,30))
        button7=wx.Button(smallPan,label="7",pos=(30,110),size=(50,30))
        button8=wx.Button(smallPan,label="8",pos=(90,110),size=(50,30))
        button9=wx.Button(smallPan,label="9",pos=(150,110),size=(50,30))
        button_star=wx.Button(smallPan,label="*",pos=(30,150),size=(50,30))
        button0=wx.Button(smallPan,label="0",pos=(90,150),size=(50,30))
        button=wx.Button(smallPan,label="#",pos=(150,150),size=(50,30))
        self.Centre()
        self.Show(True)

app = wx.App()
Border(None, -1, '')
app.MainLoop() 
