import os
from astropy.io import fits

# Укажите путь к вашему файлу
fits_filename = r"I:\AGN\OJ287\Jupyter\OJ287_86400.lc"  # Или ваше имя файла

if not os.path.exists(fits_filename):
    # Попробуем найти в текущей директории скрипта, если абсолютный путь не подошел
    fits_filename = "OJ287_86400.lc" 

# Открываем FITS-файл
with fits.open(fits_filename) as hdul:
    data_table = hdul[1].data  # явно берем первое табличное расширение
    
    # Берем время старта и стопа в секундах MET
    start_met = data_table['START']
    stop_met = data_table['STOP']
    
    # Считаем середину интервала наблюдения в MET
    mid_met = (start_met + stop_met) / 2.0
    
    # Конвертируем MET секунды в MJD по формуле Fermi-LAT
    mjd_data = (mid_met / 86400.0) + 51910.00074287037
    
    # Извлекаем поток в диапазоне 100-300000 МэВ
    flux_data = data_table['FLUX_100_300000']
    
    # Опционально: берем ошибку потока, если захотите построить график
    error_data = data_table['ERROR_100_300000']

# Сохраняем в простой текстовый файл (ДАТА в MJD и ПОТОК)
output_filename = "plain_text_lc.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    # Заголовок таблицы
    f.write("# MJD            FLUX_100_300000  ERROR_100_300000\n")
    
    # Записываем построчно
    for mjd, flux, err in zip(mjd_data, flux_data, error_data):
        f.write(f"{mjd:<15.6f} {flux:.6e} {err:.6e}\n")

print(f"Успешно! Данные сохранены в: {output_filename}")