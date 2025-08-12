# 🏥 Hospitalist Monthly Scheduler

A comprehensive Streamlit application for automating provider scheduling with customizable rules and preferences.

## ✨ Features

- **📅 Interactive Calendar**: FullCalendar-based interface for visual schedule management
- **⚙️ Flexible Rules Engine**: Global and provider-specific scheduling rules
- **👥 Provider Management**: Easy roster management with individual preferences
- **📊 Grid View**: Spreadsheet-style editing for quick assignments
- **✅ Validation**: Real-time rule violation checking
<<<<<<< HEAD
- **🔄 Auto-Generation**: Intelligent draft schedule creation from rules
=======
- **�� Auto-Generation**: Intelligent draft schedule creation from rules
>>>>>>> 8d473f28e5648cd512b8cae2150e28bf02b1192c
- **💬 Comments**: Add notes to individual shifts
- **🌐 Google Calendar Sync**: Export schedules to Google Calendar

## 🚀 Quick Start

<<<<<<< HEAD
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Open Browser**: Navigate to `http://localhost:8501`

## 📋 Usage Guide

### 1. Calendar Tab
- **Navigate**: Use year/month inputs or calendar navigation
- **Generate Draft**: Creates initial schedule based on rules
- **Validate**: Checks for rule violations
- **Edit**: Click events to modify assignments
- **Highlight**: Filter view by specific provider

### 2. Settings Tab
- **Global Rules**: Set default scheduling parameters
- **Shift Types**: Configure shift definitions and colors
- **Daily Capacities**: Set how many providers needed per shift type

### 3. Providers Tab
- **Add/Remove**: Manage provider roster
- **Individual Rules**: Set provider-specific preferences
- **Vacations**: Mark unavailable dates
- **Eligibility**: Restrict which shifts each provider can work

### 4. Grid View Tab
- **Spreadsheet Interface**: Edit assignments directly in grid
- **Apply Changes**: Sync grid edits back to calendar
- **Validation**: Automatic conflict detection

## ⚙️ Configuration

### Default Shift Types
- **N12**: 7pm–7am Night (Purple)
- **NB**: Night Bridge 11pm–7am (Cyan)
- **R12**: 7am–7pm Rounder (Green)
- **A12**: 7am–7pm Admitter (Orange)
- **A10**: 10am–10pm Admitter (Red)

### Global Rules
- **Max/Min Shifts**: Per provider limits
- **Rest Days**: Minimum rest between shifts
- **Block Size**: Preferred consecutive day assignments
- **Weekend Requirements**: Ensure weekend coverage
- **Night Limits**: Cap overnight shifts per provider

### Provider-Specific Rules
- **Shift Eligibility**: Which shift types they can work
- **Max Shifts**: Override global limits
- **Vacations**: Mark unavailable dates
- **Day/Night Ratio**: Preferred shift type balance
- **Rest Requirements**: Individual rest day preferences

## 🔧 Technical Details

- **Framework**: Streamlit
- **Calendar**: streamlit-calendar (FullCalendar.js)
- **Data Models**: Pydantic for validation
- **Persistence**: JSON files for provider rules
- **Google Integration**: Calendar API for export

## 📁 File Structure

```
IMIS/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── provider_rules.json   # Provider-specific settings (auto-created)
├── provider_caps.json    # Provider shift eligibility (auto-created)
└── IMIS_initials.csv     # Default provider roster
```

## 🐛 Troubleshooting

### Common Issues

1. **Calendar Not Loading**: Ensure `streamlit-calendar` is installed
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **Google Calendar Issues**: Check credentials.json file
4. **Performance**: Large provider rosters may slow down generation

### Debug Mode
Run with debug information:
```bash
streamlit run app.py --logger.level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

=======
### Local Development
```bash
# Clone the repository
git clone https://github.com/Alpha-nek/IMIS.git
cd IMIS

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### Streamlit Cloud Deployment
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set the main file path to: `main.py`
6. Deploy!

## 📁 Project Structure
>>>>>>> 8d473f28e5648cd512b8cae2150e28bf02b1192c
